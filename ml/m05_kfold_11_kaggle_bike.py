import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor


#1. 데이터
path = './_data/kaggle_bike/'
data= pd.read_csv(path + 'train.csv', index_col=0).dropna()

x = data.drop(['count', 'casual', 'registered'], axis = 1)
y = data['count']


n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 123)   # 데이터를 일정 비율 섞은 후 20%

#2. 모델 구성
model = RandomForestRegressor()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs= -1)


print('ACC: ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4))


# ACC:  [0.27381347 0.29543793 0.27702596 0.32968286 0.3184856 ] 
#  cross_val_score 평균 :  0.2989



