# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:38:20 2023

@author: vaish
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#now import file from data set and create a dataframe

df = pd.read_csv("C:/2-dataset/AutoInsurance.csv.xls")

a = df.describe()
