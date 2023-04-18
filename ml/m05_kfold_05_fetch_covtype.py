import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor


#1. 데이터
x, y = fetch_covtype(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state= 123, test_size=0.2
# )

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 123)   # 데이터를 일정 비율 섞은 후 20%

#2. 모델 구성
model = RandomForestRegressor()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs= -1)       # n_jobs= -1: 사용가능한 모든 CPU 코어 수


print('ACC: ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4))



# ACC:  [0.9261208  0.9263176  0.92705453 0.93073847 0.9240855 ] 
#  cross_val_score 평균 :  0.9269
