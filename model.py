import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
import pickle
df=pd.read_csv('dataset.csv')

reg=LinearRegression()
reg.fit(df[['x1','x2']], df['y'])
pickle.dump(reg, open("model.pkl", "wb"))


