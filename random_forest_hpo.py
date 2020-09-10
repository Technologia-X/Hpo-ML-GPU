import pandas as pd
import numpy as np
import math
import cupy
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from cuml.ensemble import RandomForestRegressor as curfc

import ray
from ray import tune
from ray.tune import track, trial
from ray.tune.logger import TBXLogger
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
from ray.tune.utils import get_pinned_object, pin_in_object_store

start_time = time.time()

def prepare_dataset():
    data = pd.read_csv("data.csv") #Read data file

    data.drop(['Unnamed: 0', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',
            'Potential', 'Club', 'Club Logo', 'Wage', 'Special',
            'Preferred Foot', 'International Reputation', 'Weak Foot', 
            'Body Type', 'Real Face',
            'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
            'Height', 'Weight', 'LS', 'ST', 'RS', 'LW',
            'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
            'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause'], axis=1, inplace=True) #Drop unused columns

    data.dropna(inplace=True) #drop na rows
    data.reset_index(inplace=True, drop=True) # reset the index of dataframe
    data['Value'] = data['Value'].str.replace('€','') # remove euro symbol from value (dependent var)
    data['Value'] = data['Value'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int) #convert into million value and int

    pos = pd.get_dummies(data['Position']) #one hot encoding position column
    work_rate = pd.get_dummies(data['Work Rate']) #one hot encoding work_rate column

    data = pd.concat([data, pos], axis=1) #Concat pos one hot encoding with original dataframe
    data = pd.concat([data, work_rate], axis=1) #Concat work rate one hot encoding with original dataframe

    del data['Position']
    del data['Work Rate'] #Delete both object dtyped columns

    return data

trial_num = 0

def get_trial_name(trial: ray.tune.trial.Trial):
    # Returns the trial number over an iterator variable trail_num
    global trial_num
    trial_num = trial_num + 1
    trial_name = trial.trainable_name + "_" + str(trial_num)
    return trial_name

class PerfTimer:
    # High resolution timer for reporting training and inference time.
    def __init__(self):
        self.start = None
        self.duration = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration = time.perf_counter() - self.start

class BaseTrainTransformer(tune.Trainable):
    @property
    def static_config(self) -> dict:
        return getattr(self, "_static_config", {})

    def _setup(self, config: dict):

        self._dataset, self._col_labels, self._y_label = (
            get_pinned_object(data_id),
            None,
            "Value",
        )
        
        self.rf_model = None
        self._build(config)

    def _build(self, new_config):
        self._model_params = {
            "max_depth": int(new_config["max_depth"]),
            "n_estimators": int(new_config["n_estimators"]),
            "max_features": float(new_config["max_features"]),
            "n_bins": 16,  # args.n_bins,
            "seed": time.time(),
        }
        self._global_best_model = None
        self._global_best_test_accuracy = 0

    def _train(self):
        iteration = getattr(self, "iteration", 0)
        # print(self._dataset)
        if compute == "GPU":
            # split data
            X_train, X_test, y_train, y_test = train_test_split(
                self._dataset.loc[:, self._dataset.columns != self._y_label],
                self._dataset[self._y_label],
                train_size=0.8,
                shuffle=False,
                random_state=iteration,
            )
            self.rf_model = curfc(
                n_estimators=self._model_params["n_estimators"],
                max_depth=self._model_params["max_depth"],
                n_bins=self._model_params["n_bins"],
                max_features=self._model_params["max_features"],
            )
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')

        # train model
        with PerfTimer() as train_timer:
            trained_model = self.rf_model.fit(X_train, y_train.astype("float32"))
        training_time = train_timer.duration

        # evaluate perf
        with PerfTimer() as inference_timer:
            test_accuracy = r2_score(y_test.astype("float32"), trained_model.predict(X_test))
        infer_time = inference_timer.duration

        # update best model [ assumes maximization of perf metric ]
        if test_accuracy > self._global_best_test_accuracy:
            self._global_best_test_accuracy = test_accuracy
            self._global_best_model = trained_model

        return {
            "test_accuracy": test_accuracy,
            "train_time": round(training_time, 4),
            "infer_time": round(infer_time, 4),
            "is_bad": not math.isfinite(test_accuracy),
        }

    def _save(self, checkpoint_dir):
        return {
            "test_accuracy": self._global_best_test_accuracy,
        }

    def _restore(self, checkpoint):
        self._global_best_test_accuracy = checkpoint["test_accuracy"]

    def reset_config(self, new_config):
        # Rebuild the config dependent stuff
        del self.rf_model
        self._build(new_config)
        self.config = new_config
        return True