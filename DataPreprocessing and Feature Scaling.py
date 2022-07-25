# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:50:37 2022

@author: Asim Pardeshi
"""
#IMPORT THE  PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORT DATASET
dataset=pd.read_csv(r'D:\GCLASSROOM\12july\Data\Data.csv')

#SPLIT DATASETVINTO INDEPENDENT AND DEPENDENT VARIABLE
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#IMPUTE NUMERIAL VALUES
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="median")
imputer = SimpleImputer()
imputer=imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3]) #REPLACES MISSING VALUES WITH MEDIAN STRATEGY

#IMPUTE CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])
 #we need to split dataset into training and testing phase
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#scale the data--feature scalinhg
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
