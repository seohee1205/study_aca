# Halving

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time


#1. 데이터
# x, y = load_iris(return_X_y=True)
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234, test_size= 0.2, 
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
# model = GridSearchCV(SVC(), parameters, 
# model = RandomizedSearchCV(SVC(), parameters, 
model = HalvingGridSearchCV(SVC(), parameters, 
                    #  cv=kfold, 
                     cv=5,      # 분류의 디폴트는 StratifiedKFold야
                     verbose=1, 
                     refit= True,      # 디폴트값은 True, False하게 되면 최종 파라미터로 출력
                    # refit=False,       # True는 최상의 파라미터로 출력 
                    #  n_iter=5,      # 랜덤서치의 파라미터라 안 됨
                     factor=3.2,      # 디폴트= 3임 
                     n_jobs= -1)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("걸린시간 : ", round(end_time - start_time, 2), '초')
# 로드디지트 그리드서치 걸린시간 : 8.74 초
# 로드디지트 할빙그리드서치 걸린시간 :  3.71 초

# print(x.shape, x_train.shape)   #(1797, 64) (1437, 64)
'''
n_iterations: 3                     # 3번 전체 훈련
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100                 # 최소 훈련데이터 개수
max_resources_: 1437                # 최대 훈련데이터 개수
aggressive_elimination: False
factor: 3                   # 요인, 인자 n빵
----------
iter: 0
n_candidates: 52        # 전체 파라미터의 개수
n_resources: 100        # 0번째 훈련때 쓸 훈련데이터 개수
Fitting 5 folds for each of 52 candidates, totalling 260 fits
----------
iter: 1
n_candidates: 18        # 전체 파라미터 개수 / factor => 52(candidates)를 3(factor)으로 나눔  -> 상위 18개만 사용하겠다 
n_resources: 300        # min_resources * factor -> resources * 3(factor)     -> 데이터의 개수는 3배로 늘림
Fitting 5 folds for each of 18 candidates, totalling 90 fits      # 300개의 데잍로 5 * 18 = 90번 훈련하겠다 
----------
iter: 2
n_candidates: 6         # 18 / 3(factor)
n_resources: 900        # 300 * 3(faactor)
Fitting 5 folds for each of 6 candidates, totalling 30 fits        # 900개의 데이터로 5 * 6 = 30번 훈련하겠다
'''

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
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # ascending = True 오름차순이 default
# print(pd.DataFrame(model.cv_results_).columns)  # 컬럼 확인

path = './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm14_HalvingGridSearch1.csv')
# \ : 줄바꿈


# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9888888888888889
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  2.5 초
