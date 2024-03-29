import os
import pandas as pd
import numpy as np
from read_file import load_min_distance_sample
from preprocessing import split_month_day_hour
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from prepare_train_test_data import save_concated_data_min_distance
from preprocessing import Imputation
from read_file import load_min_distance
import time

def split_month_day_hour(DataFrame:pd.DataFrame)->pd.DataFrame:
    month_date_time_min=[i.split(' ') for i in DataFrame['일시']]
    DataFrame=DataFrame.drop(['연도','일시'],axis=1)
    month_date=[j.split('-')for j in [i[0] for i in month_date_time_min]]
    time_min=[j.split(':')for j in[i[1] for i in month_date_time_min]]
    month=pd.Series([float(i[0]) for i in month_date],name='월', index=DataFrame.index)
    date=pd.Series([float(i[1]) for i in month_date],name='일', index=DataFrame.index)
    time=pd.Series([float(i[0])for i in time_min],name='시', index=DataFrame.index)
    DataFrame=pd.concat([month,date,time,DataFrame],axis=1)
    return DataFrame

sta = time.time()
answer_sample=pd.read_csv('d:/study_data/_data/aif/초미세먼지/answer_sample.csv')
print(answer_sample)
train_datas,test_datas,pmname=load_min_distance()
test_datas=pd.concat(test_datas,axis=0)
missing_indices = np.where(pd.isnull(test_datas['PM2.5'].values))[0]
print(len(missing_indices))
print(test_datas)

test_datas=Imputation(split_month_day_hour(test_datas))

PM25=test_datas['PM2.5']
answer_sample['PM2.5']=np.round(PM25[missing_indices].values, 6)

end = time.time()
print('걸린 시간 :', np.round(end-sta,2),'초')

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
save_path = 'd:/study_data/_save/aif/초미세먼지/'

answer_sample.to_csv(save_path + date + '_submission.csv', index=False)
