'''
This file is called for web_app.py
'''

import os
import sys
import streamlit as st
from flask import Flask
import joblib

import config as cf
from src.util import data_util as du
from src.model import house_price as hp


app = Flask(__name__)
st.write("test page")
# load raw data file
st.write('Raw data sample:')
df = du.read_local_file(cf.DATA_RAW_PATH, cf.house_price_data_file, 'csv')
st.write(df.shape)
st.write(df.head())

# train data, test data
train_df = du.read_local_file(cf.DATA_PROCESSED_PATH, cf.train_house_price, 'csv')
test_df = du.read_local_file(cf.DATA_PROCESSED_PATH, cf.test_house_price, 'csv')
st.write('Train data size: ', train_df.shape)
st.write(train_df.head())
st.write()
st.write('Test data size: ', test_df.shape)
st.write(test_df.head())

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

# Main function
if __name__ == '__main__':
    app.debug = False
    #port = int(os.environ.get('PORT', 5000))
    app.run()





