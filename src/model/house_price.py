# utitlity libraries
import sys
import os
from io import BytesIO
# to persist the model and the scaler
import joblib
import streamlit as st

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd

# Regression
#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# user-defined functions
import config as cf
from src.util import data_util as du

# Define variables - define in config.yml file
TARGET = 'price'
TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                  'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                  'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
                  'zipcode', 'season_spring', 'season_summer', 'season_winter']

# define path to store model result: pickle file, train_vars, scaling, define transformed data

# define functions for data cleaning/processing data/ feature engineering in class
# call function for prediction

class HousePrice:
    
    ######################################### Define variables used in class ####################################
    def __init__(self):
        self.model = None
        self.train_pred = None
        self.test_pred = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        
    def prepare_dataset(self, train_data=None, test_data=None):
        train_df = du.read_local_file(cf.DATA_PROCESSED_PATH, cf.train_house_price, 'csv')
        test_df = du.read_local_file(cf.DATA_PROCESSED_PATH, cf.test_house_price, 'csv')
        self.X_train = train_df[TRAIN_VARS]
        self.X_test = test_df[TRAIN_VARS]
        self.y_train = train_df[TARGET]
        self.y_test = test_df[TARGET]
    
    
    
    
    def training(self):
        model_file = cf.house_price_model
        model = joblib.load(model_file)

        # prediction
        self.train_pred = model.predict(self.X_train)
        self.test_pred = model.predict(self.X_test)
        
        # Model performance
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        
        print('Train score: ', train_score)
        print('Test score: ', test_score)        

        
        
        
        
    
        
    


