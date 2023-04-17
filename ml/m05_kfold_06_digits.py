import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor


#1. 데이터
x, y = load_digits(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state= 123, test_size=0.2
# )

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 123)   # 데이터를 일정 비율 섞은 후 20%

#2. 모델 구성
model = RandomForestRegressor()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs= -1)


print('ACC: ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4))


# ACC:  [0.84060287 0.89784759 0.85602482 0.87834414 0.88606982] 
#  cross_val_score 평균 :  0.8718

