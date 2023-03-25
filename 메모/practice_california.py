import numpy as np
import pandas as pd 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, GRU, Input, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = 