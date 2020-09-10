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