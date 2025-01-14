# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:45:50 2025

@author: Mai
"""

import sys
import os
import pandas as pd

import config as cf

# read a csv file
def read_local_file(file_path=None, file_name=None, type=None):
    if type == 'csv':
        data = pd.read_csv(file_path + file_name)       
    
    return data



def read_s3_file(bucket_name=None, file_path=None, file_name=None, type=None):
    if type == 'local':
        data = pd.read_csv('data/' + file_name)
        
    return data