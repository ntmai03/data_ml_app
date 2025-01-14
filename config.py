# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:41:45 2025

@author: Mai
"""

import streamlit as st
import sys
import os
from pathlib import Path
#import boto3
import yaml


# Define path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_PATH = 'data/raw/'
DATA_PROCESSED_PATH = 'data/processed/'

MODEL_PATH = 'model/'


# data file
house_price_data_file = 'kc_house_data.csv'
train_house_price = 'houseprice_train.csv'
test_house_price = 'houseprice_test.csv'

# model
house_price_model = MODEL_PATH + 'house_price_gbt.pkl'
