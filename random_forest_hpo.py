import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import numpy as np
import math
import cupy
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from cuml.ensemble import RandomForestRegressor as curfc
import cudf as cd

import ray
from ray import tune
from ray.tune import track, trial
from ray.tune.logger import TBXLogger
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
from ray.tune.utils import get_pinned_object, pin_in_object_store

data = cd.read_csv("data.csv")
data.drop(['﻿', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',
           'Potential', 'Club', 'Club Logo', 'Wage', 'Special',
           'Preferred Foot', 'International Reputation', 'Weak Foot', 
           'Body Type', 'Real Face', 'Position',
           'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
           'Height', 'Weight'], inplace=True)
print(data.columns)
print(data.info(verbose=True))