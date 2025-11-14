# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:25:00 2023

@author: h_min
"""

import pandas as pd 

input_dir = "C:/Users/h_min/ML_Demand/"
fname = 'salty_snack_temp_0.05.csv'

data = pd.read_csv(input_dir + fname)

data['iri_key'].nunique()


data['week'].nunique()


data['colupc'].nunique()