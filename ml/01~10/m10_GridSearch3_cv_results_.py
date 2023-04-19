# 그물망처럼 찾겠다
# GridSearch에서 분류의 디폴트는 StratifiedKFold야

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import pandas as pd

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size= 0.2, 
    # stratify=y
)

n_splits= 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":['linear'], "degree":[3, 4, 5]},      #12
    {'C':[1, 10, 100], 'kernel':['rbf', 'linear'], 'gamma':[0.001, 0.0001]},    # 12
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'],
     'gamma':[0.01, 0.001, 0.0001], 'degree':[3, 4]},       # 24
    {'C':[0.1, 1], 'gamma':[1, 10]}
]       # 총 52번 

#2. 모델
model = GridSearchCV(SVC(), parameters, 
                    #  cv=kfold, 
                     cv=5,      # 분류의 디폴트는 StratifiedKFold야
                     verbose=1, 
                     refit= True,      # 디폴트값은 True, False하게 되면 최종 파라미터로 출력
                    # refit=False,       # True는 최상의 파라미터로 출력 
                     n_jobs= -1)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9916666666666668
# model.score :  1.0

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# accuracy_score :  1.0

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
# 최적 튠 ACC :  1.0

print("걸린시간 : ", round(end_time - start_time, 2), '초')
# 걸린시간 :  2.34 초

#######################################################################
# print(pd.DataFrame(model.cv_results_))
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # ascending = True 오름차순이 default
print(pd.DataFrame(model.cv_results_).columns)  # 컬럼 확인

path = './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm10_GridSearch3.csv')
# \ : 줄바꿈
