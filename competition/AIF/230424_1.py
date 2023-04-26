import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import datetime

path = 'd:/study_data/_data/aif/초미세먼지/'
save_path = 'd:/study_data/_save/aif/초미세먼지/'

train_path = 'd:/study_data/_data/aif/초미세먼지/TRAIN/'
train_AWS_path = 'd:/study_data/_data/aif/초미세먼지/TRAIN_AWS/'
test_INPUT_path = 'd:/study_data/_data/aif/초미세먼지/TEST_INPUT/'
test_AWS_path = 'd:/study_data/_data/aif/초미세먼지/TEST_AWS/'
meta_path = 'd:/study_data/_data/aif/초미세먼지/META/'

file_list = []