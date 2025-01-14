import os
import sys
import config as cf
import joblib

from src.util import data_util as du
from src.model import house_price as hp

# load raw data file
print('Raw data sample:')
df = du.read_local_file(cf.DATA_RAW_PATH, cf.house_price_data_file, 'csv')
print(df.shape)
print(df.head())

# train data, test data
train_df = du.read_local_file(cf.DATA_PROCESSED_PATH, cf.train_house_price, 'csv')
test_df = du.read_local_file(cf.DATA_PROCESSED_PATH, cf.test_house_price, 'csv')
print('Train data size: ', train_df.shape)
print(train_df.head())
print()
print('Test data size: ', test_df.shape)
print(test_df.head())

# Define variables
TARGET = 'price'
TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                  'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                  'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
                  'zipcode', 'season_spring', 'season_summer', 'season_winter']
X_train = train_df[TRAIN_VARS]
X_test = test_df[TRAIN_VARS]
y_train = train_df[TARGET]
y_test = test_df[TARGET]

# Data Processing

# Train model
        
# call model
model_file = cf.house_price_model
model = joblib.load(model_file)

# prediction
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
        
# Model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)        
print('Train score: ', train_score)
print('Test score: ', test_score)  



# call class
house_price =  hp.HousePrice()
house_price.prepare_dataset()
print('Train data size: ', house_price.X_train.shape)
print(house_price.X_train.head())
print()
print('Test data size: ', house_price.X_test.shape)
print(house_price.X_train.head())
print('Model performance')
house_price.training()
 


