import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score


# / = // = \ = \\   모두 같다
# \a => text파일에서 하면 띄어쓰기, /n => 줄바꿈  (노란색으로 바뀜)

path = 'd:/study_data/_data/aif/초미세먼지/'
TRAIN = 'd:/study_data/_data/aif/초미세먼지/TRAIN/'
TRAIN_AWS = 'd:/study_data/_data/aif/초미세먼지/TRAIN_AWS/'
TEST_INPUT = 'd:/study_data/_data/aif/초미세먼지/TEST_INPUT/'
TEST_AWS = 'd:/study_data/_data/aif/초미세먼지/TEST_AWS/'
META = 'd:/study_data/_data/aif/초미세먼지/META/'
answer = 'd:/study_data/_data/aif/초미세먼지/META/answer_sample.csv'


train_files = glob.glob(path + "TRAIN/*.csv")
# print(train_files)
test_input_files = glob.glob(path + "test_input/*.csv")
# print(test_input_files)

######################################## TRAIN 폴더 ###########################################
list = []
for filename in train_files:
    df = pd.read_csv(filename, index_col = None, header=0,
                     encoding = 'utf-8-sig')    # 한글 깨짐

    list.append(df)     # 이어짐
    
# print(list)         # [35064 rows x 4 columns]]   * 17개
# print(len(list))    # 17

# 리스트들을 이어줌
train_dataset = pd.concat(list, axis = 0, 
                          ignore_index= True)   # 새로운 index 생성

# print(train_dataset)    # [596088 rows x 4 columns]

######################################## TEST_INPUT 폴더 ###########################################
list = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col = None, header=0,
                     encoding = 'utf-8-sig')    # 한글 깨짐

    list.append(df)     # 이어짐
    
# print(list)         # [7728 rows x 4 columns]]   * 17개
# print(len(list))    # 17

# 리스트들을 이어줌
test_input_dataset = pd.concat(list, axis = 0, 
                          ignore_index= True)   # 새로운 index 생성

# print(test_input_dataset)    # [131376 rows x 4 columns]

######################################## 측정소 지역 라벨인코더 ###########################################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])          # 새로운 메모리 공간 locate 새로 만듦
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])     # 데이터의 위치나 값이 바뀔 수 있기 때문에 train_fit 그대로 사용(fit X)
# print(train_dataset)        # [596088 rows x 5 columns]
# print(test_input_dataset)   # [131376 rows x 5 columns]

train_dataset = train_dataset.drop(['측정소'], axis = 1)                 # axis = 0 -> 행삭제, 1 -> 열삭제
test_input_dataset = test_input_dataset.drop(['측정소'], axis = 1)       # axis = 0 -> 행삭제, 1 -> 열삭제
# print(train_dataset)        # [596088 rows x 4 columns]
# print(test_input_dataset)   # [131376 rows x 4 columns]

######################################## 일시 컬럼을 월, 일, 시간으로 분리 ###########################################
# ex) 12_31 21:00  ->  12와 21 추출
# print(train_dataset.info())

############# train 변경
train_dataset['month'] = train_dataset['일시'].str[:2]
# print(train_dataset['month'])
train_dataset['hour'] = train_dataset['일시'].str[6:8]
# print(train_dataset['hour'])
# print(train_dataset)    # [596088 rows x 6 columns]

train_dataset = train_dataset.drop(['일시'], axis = 1)                 # axis = 0 -> 행삭제, 1 -> 열삭제
# print(train_dataset)    # [596088 rows x 5 columns]

### str -> int 
# train_dataset['month'] = pd.to_numeric(train_dataset['month'])     #  pd.to_numeric : str을 수치형 데이터로 바꿔줌
# train_dataset['month'] = pd.to_numeric(train_dataset['month']).astype('int8')
train_dataset['month'] = train_dataset['month'].astype('int8')
train_dataset['hour'] = train_dataset['hour'].astype('int8')
# print(train_dataset.info())


############ test_input 변경
test_input_dataset['month'] = test_input_dataset['일시'].str[:2]
# print(test_input_dataset['month'])
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8]
# print(test_input_dataset['hour'])
# print(test_input_dataset)    # [131376 rows x 6 columns]

test_input_dataset = test_input_dataset.drop(['일시'], axis = 1)                 # axis = 0 -> 행삭제, 1 -> 열삭제
# print(test_input_dataset)    # [131376 rows x 5 columns]

### str -> int 
# test_input_dataset['month'] = pd.to_numeric(test_input_dataset['month'])     #  pd.to_numeric : str을 수치형 데이터로 바꿔줌
# test_input_dataset['month'] = pd.to_numeric(test_input_dataset['month']).astype('int8')
test_input_dataset['month'] = test_input_dataset['month'].astype('int8')
test_input_dataset['hour'] = test_input_dataset['hour'].astype('int8')
print(test_input_dataset.info())


######################################## 결측치 제거 PM2.5에 15542개 있음 ###########################################
# 결측치 확인
# print(train_dataset.info())    # 결측치 있음
# 전체 596085 -> 580546 으로 줄인다.
train_dataset = train_dataset.dropna() 
# print(train_dataset.info())
#  0   연도      580546 non-null  int64
#  1   PM2.5   580546 non-null  float64
#  2   locate  580546 non-null  int32
#  3   month   580546 non-null  int8
#  4   hour    580546 non-null  int8

#### 시즌 -> 파생피처도 생각해봐 ####

# y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis = 1)
y = train_dataset['PM2.5']
print(x, '\n', y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 3377, shuffle= True
)

parameters =  {'n_estimators' : 10000,               # = epochs
              'learning_rate' : 0.3,
              'max_depth' : 5,
              'gamma' : 0,
            #   'min_child_weight' : 1,
            #   'subsample' : 1,                     # dropout
            #   'colsample_bytree' : 1,
            #   'colsample_bylevel' : 1,
            #   'colsample_bynode' : 1,
            #   'reg_alpha' : 0,                    # 절대값: 레이어에서 양수만들겠다/ 라쏘 / 머신러닝 모델
            #   'reg_lambda' : 1,                   # 제곱: 레이어에서 양수만들겠다/ 리지   / 머신러닝 모델
            #   'random_state' : 3377,
              'n_jobs' : -1
}


#2. 모델구성
model = XGBRegressor()

#3. 컴파일, 훈련
model.set_params(**parameters,
                 eval_metric = 'mae',
                 early_stopping_rounds = 300,
                 )

start_time = time.time()
model.fit(x_train, y_train, verbose= 1,
          eval_set = [(x_train, y_train), (x_test, y_test)]
)
end_time = time.time()
print("걸린시간 : ", round(end_time - start_time, 2), "초")

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.score(x_test, y_test)
print("model.score : ", results)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

mae = mean_absolute_error(y_test, y_predict)
print("mae 스코어 : ", mae)

# 걸린시간 :  75.22 초
# model.score :  0.2587066479646838
# r2 스코어 :  0.2587066479646838
# mae 스코어 :  0.043393791461033254

# 결측치만 추출
# test_input_dataset에서 결측치만 추출해서 데이터셋 만듦

missing_rows = test_input_dataset[test_input_dataset.isnull().any(axis=1)]
print(missing_rows.shape)#(78336, 5)
print(missing_rows.info())

x_test_miss = missing_rows.drop(['PM2.5'], axis=1)

y_submit = model.predict(x_test_miss)
# print(y_submit)

submission = pd.read_csv(path+'answer_sample.csv',index_col=None, header=0, 
                     encoding = 'utf-8-sig')
                    
print(submission)
submission['PM2.5'] = y_submit
print(submission)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
save_path = 'd:/study_data/_save/aif/초미세먼지/'

submission.to_csv(save_path + date + '_submission.csv', index=False)
