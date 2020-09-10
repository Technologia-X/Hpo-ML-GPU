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

#At this point all columns should be either int or float

# Break the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Value'], axis=1), data['Value'], train_size=0.8, shuffle=True,random_state=42)

X_train = X_train.astype('float32') # Convert to float32 as rapids cuml requires float32 data type for training
X_test = X_test.astype('float32')

cuml_model = curfc() # Define random forest regressor model with default settings (no HPO)
cuml_model.fit(X_train,y_train) # Train the model on training set

preds = cuml_model.predict(X_test) #Predict the test set

# Metrics for accuracy validation
print(r2_score(y_test, preds))
print(mean_absolute_error(y_test, preds))
print(mean_squared_error(y_test, preds))