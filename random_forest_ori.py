import pandas as pd
import numpy as np
import math
import cupy
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from cuml.ensemble import RandomForestRegressor as curfc
import cudf as cd

start_time = time.time()

data = cd.read_csv("data.csv")
data.drop(['﻿', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',
           'Potential', 'Club', 'Club Logo', 'Wage', 'Special',
           'Preferred Foot', 'International Reputation', 'Weak Foot', 
           'Body Type', 'Real Face', 'Position',
           'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
           'Height', 'Weight', 'LS', 'ST', 'RS', 'LW',
           'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
           'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',], inplace=True)

print(data.head(10))
print(data.info(verbose=True))