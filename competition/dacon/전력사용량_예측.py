import random
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings(action='ignore') 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train_df = pd.read_csv('d:/_study_data/_data/전력사용량/train.csv')
test_df = pd.read_csv('d:/_study_data/_data/전력사용량/test.csv')

#결측값을 0으로 채웁니다
train_df = train_df.fillna(0)

#시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
train_df['month'] = train_df['일시'].apply(lambda x : int(x[4:6]))
train_df['day'] = train_df['일시'].apply(lambda x : int(x[6:8]))
train_df['time'] = train_df['일시'].apply(lambda x : int(x[9:11]))

train_x = train_df.drop(columns=['num_date_time', '일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']

model = RandomForestRegressor()
model.fit(train_x, train_y)

test_df['month'] = test_df['일시'].apply(lambda x : int(x[4:6]))
test_df['day'] = test_df['일시'].apply(lambda x : int(x[6:8]))
test_df['time'] = test_df['일시'].apply(lambda x : int(x[9:11]))

test_x = test_df.drop(columns=['num_date_time', '일시'])

preds = model.predict(test_x)

submission = pd.read_csv('d:/_study_data/_data/전력사용량/sample_submission.csv')
submission

submission['answer'] = preds
submission

submission.to_csv('d:/_study_data/_save/전력사용량/submission.csv', index=False)