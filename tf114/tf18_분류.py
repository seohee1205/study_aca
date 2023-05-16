import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, fetch_covtype, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
tf.set_random_seed(337)

#1. 데이터 
data_list = [load_iris(return_X_y=True), load_diabetes(return_X_y=True), fetch_covtype(return_X_y=True), load_wine(return_X_y=True)]
random_state= 337

for i in range(len(data_list)):
    data = data_list[i]
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = random_state, shuffle = True)
    