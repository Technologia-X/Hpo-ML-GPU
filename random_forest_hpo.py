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
                shuffle=True,
                random_state=42,
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

class WrappedTrainable(BaseTrainTransformer):
    def __init__(self, *args, **kwargs):

        self._static_config = static_config

        super().__init__(*args, **kwargs)

def build_search_alg(search_alg, param_ranges: dict):
    """
    Initialize a search algorithm that is selected using 'search_alg'
    
    Parameters
    ----------
        search_alg   : str; Selecting the search algorithm. Possible values
                       [BayesOpt, SkOpt]
        param_ranges : dictionary of parameter ranges over which the search
                       should be performed

    Returns
    -------
        alg : Object of the RayTune search algorithm selected
    """

    alg = None

    if search_alg == "BayesOpt":
        from ray.tune.suggest.bayesopt import BayesOptSearch

        alg = BayesOptSearch(
            param_ranges,
            max_concurrent=max_concurrent,
            metric="test_accuracy",
            mode="max",
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
        )

    elif search_alg == "SkOpt":

        from skopt import Optimizer
        from skopt.space import Real, Integer
        from ray.tune.suggest.skopt import SkOptSearch

        opt_params = [
            Integer(param_ranges["n_estimators"][0], param_ranges["n_estimators"][1]),
            Integer(param_ranges["max_depth"][0], param_ranges["max_depth"][1]),
            Real(
                param_ranges["max_features"][0],
                param_ranges["max_features"][1],
                prior="log-uniform",
            ),
        ]

        optimizer = Optimizer(opt_params)

        alg = SkOptSearch(
            optimizer,
            list(param_ranges.keys()),
            max_concurrent=max_concurrent,
            metric="test_accuracy",
            mode="max",
        )
    else:
        print("Unknown Option. Select BayesOpt or SkOpt")
    return alg

def select_sched_alg(sched_alg):
    """
     Initialize a scheduling algorithm that is selected using 'sched_alg'
    
    Parameters
    ----------
        sched_alg   : str; Selecting the search algorithm. Possible values
                       [MedianStop, AsyncHyperBand]

    Returns
    -------
        alg : Object of the RayTune scheduling algorithm selected
    """
    sched = None
    if sched_alg == "AsyncHyperBand":
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="test_accuracy",
            mode="max",
            max_t=50,
            grace_period=1,
            reduction_factor=3,
            brackets=3,
        )
    elif sched_alg == "MedianStop":
        sched = MedianStoppingRule(
            time_attr="time_total_s",
            metric="test_accuracy",
            mode="max",
            grace_period=1,
            min_samples_required=3,
        )
    else:
        print("Unknown Option. Select MedianStop or AsyncHyperBand")
    return sched

num_samples = 10 #times to run the HPO, so 10 here means it will run it 10 * CV_folds
compute = (
    "GPU"  # Can take a CPU value (only for performance comparison. Not recommended)
)
CV_folds = 3 # The number of Cross-Validation folds to be performed
search_alg = "BayesOpt"  # Options: SkOpt or BayesOpt
sched_alg = "AsyncHyperBand"  # Options: AsyncHyperBand or MedianStop

# HPO Param ranges
# NOTE: Depending on the GPU memory we might need to adjust the parameter range for a successful run (I am only using GTX 1060 so I am limited)
n_estimators_range = (500, 1500)
max_depth_range = (10, 20)
max_features_range = (0.5, 1.0)

# hpo range defined to be pumped into hpo method
hpo_ranges = {
    "n_estimators": n_estimators_range,
    "max_depth": max_depth_range,
    "max_features": max_features_range,
}

ray.init(memory=11000 * 1024 * 1024, object_store_memory= 500 * 1024 * 1024, driver_object_store_memory= 100 * 1024 * 1024, local_mode=False, num_gpus=1)

max_concurrent = cupy.cuda.runtime.getDeviceCount()

cdf = prepare_dataset() #prepare dataset 

# for shared access across processes
data_id = pin_in_object_store(cdf)
    
search = build_search_alg(search_alg, hpo_ranges)

sched = select_sched_alg(sched_alg)

exp_name = None

if exp_name is not None:
    exp_name += exp_name
else:
    exp_name = ""
    exp_name += "{}_{}_CV-{}_{}M_SAMP-{}".format(
        "RF", compute, CV_folds, int(len(cdf) / 1000000), num_samples
    )

    exp_name += "_{}".format("Random" if search_alg is None else search_alg)

    if sched_alg is not None:
        exp_name += "_{}".format(sched_alg)

static_config = {
    "num_workers": 1,
}

print("Model HPO running")

analysis = tune.run(
    WrappedTrainable,
    name=exp_name,
    scheduler=sched,
    search_alg=search,
    stop={"training_iteration": CV_folds, "is_bad": True,},
    resources_per_trial={"cpu": 0, "gpu": 1},
    num_samples=num_samples,
    checkpoint_at_end=True,
    keep_checkpoints_num=1,
    local_dir="./results",
    trial_name_creator=get_trial_name,
    checkpoint_score_attr="test_accuracy",
    config={
        "n_estimators": tune.randint(n_estimators_range[0], n_estimators_range[1]),
        "max_depth": tune.randint(max_depth_range[0], max_depth_range[1]),
        "max_features": tune.loguniform(max_features_range[0], max_features_range[1]),
    },
    verbose=2,
    raise_on_failed_trial=False,
)
print("Time taken", time.time() - start_time)
analysis.dataframe().to_csv("trials.csv", index=False)