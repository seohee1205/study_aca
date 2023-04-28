import numpy as np
import pandas as pd



#1. 데이터
path = 'd:/study_data/_data/dacon_항공/'
path_save = 'd:/study_data/_save/dacon_항공/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

print(train_csv.shape)  # (1000000, 18)
print(test_csv.shape)   # (1000000, 17)

#1-1. 결측치 확인
print(train_csv.isnull().sum())

# 결측치 처리

#1-3. x, y 분리
x = train_csv.dro[]






