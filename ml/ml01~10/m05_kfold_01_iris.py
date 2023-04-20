import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



#1. 데이터
x, y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state= 123, test_size=0.2
# )

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 123)   # 데이터를 일정 비율 섞은 후 20% / shuffle 기본값: False

#2. 모델 구성
model = LinearSVC()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
# print(scores)       # [0.96666667 1.         0.93333333 0.93333333 0.9       ]
# scores = cross_val_score(model, x, y, cv=5)


print('ACC: ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4))
# ACC:  [0.96666667 1.         0.93333333 0.93333333 0.9       ]
# cross_val_score 평균 :  0.9467




