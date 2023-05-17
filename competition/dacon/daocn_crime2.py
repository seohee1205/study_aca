import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

#0. fix seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

#1. 데이터
path = 'd:/study_data/_data/dacon_crime/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

x = train_csv.drop(['TARGET'], axis = 1)
y = train_csv['TARGET']


# 범주형 변수 리스트
qual_col = ['요일', '범죄발생지']

# 원-핫 인코딩
x = pd.get_dummies(x, columns=qual_col)
test_csv = pd.get_dummies(test_csv, columns=qual_col)

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 377
)

from sklearn.model_selection import RandomizedSearchCV

# # 파라미터 그리드 설정
# param_grid = {
#     'iterations': [1000, 2000, 5000],
#     'depth': [6, 8, 10],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'l2_leaf_reg': [0.01, 0.1, 1.0],
#     'one_hot_max_size': [16, 32, 64],
#     'random_strength': [0.01, 0.1, 1.0],
#     'bagging_temperature': [0.8, 0.9, 1.0],
#     'border_count': [100, 200, 300]
# }

param_grid = {
    'iterations': [1000, 3000],
    'depth': [12, 16],
    'learning_rate': [0.0001],
    'l2_leaf_reg': [0.1],
    'one_hot_max_size': [180],
    'random_strength': [0.01],
    'bagging_temperature': [1.0],
    'border_count': [200],
}


# 랜덤 탐색을 통한 파라미터 최적화
model = CatBoostClassifier()

random_search = RandomizedSearchCV(
    model,
    param_grid,
    n_iter=2,  # 탐색할 파라미터 조합의 개수
    scoring='f1_macro',
    cv=5,  # 교차 검증 폴드 수
    random_state= 42
)

random_search.fit(x_train, y_train)

# 최적의 파라미터 출력
# print("Best parameters:", random_search.best_params_)

# 최적의 모델 사용
# best_model = random_search.best_estimator_
# random_search.best_estimator_.set_params(early_stopping_rounds=10, **param_grid)

# 최적의 모델 사용
best_model = random_search.best_estimator_
best_model.fit(x_train, y_train)  # 모델 학습

y_predict= best_model.predict(x_test)
f1 = f1_score(y_test, y_predict, average='macro')
print("f1score : ", f1)

#time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# Submission
save_path = 'd:/study_data/_save/dacon_crime/'
y_submit= best_model.predict(test_csv)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]]= y_submit
sample_submission_csv.to_csv(save_path + 'crime_' + date + '.csv', index=False)
