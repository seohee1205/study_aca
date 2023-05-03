bayesian_params = {
    'max_depth' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50),
}

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드 및 전처리
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bayesian Optimization을 위한 목적함수 정의
def lgbm_evaluate(num_leaves, max_depth, learning_rate, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda):
    # 모델 정의
    model = LGBMRegressor(
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        min_child_samples=int(min_child_samples),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42
    )
    # 모델 학습
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
    # 예측
    y_pred = model.predict(X_test)
    # 평가
    r2 = r2_score(y_test, y_pred)
    # 반환
    return r2

# Bayesian Optimization을 위한 하이퍼파라미터 범위 정의
bayesian_params = {
    'max_depth' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50),
}


# Bayesian Optimization 실행
optimizer = BayesianOptimization(
    f=lgbm_evaluate,
    pbounds=bayesian_params,
    random_state=42,
    verbose=2
)
optimizer.maximize(n_iter=10)

# 최적 파라미터 출력
print(optimizer.max)

# 최적 파라미터로 모델 재학습 및 예측
model = LGBMRegressor(
    num_leaves=int(optimizer.max['params']['num_leaves']),
    max_depth=int(optimizer.max['params']['max_depth']),
    learning_rate=optimizer.max['params']['learning_rate'],
    min_child_samples=int(optimizer.max['params']['min_child_samples']),
    subsample=optimizer.max['params']['subsample'],
    colsample_bytree=optimizer.max['params']['colsample_bytree'],
    reg_alpha=optimizer.max['params']['reg_alpha'],
    reg_lambda=optimizer.max['params']['reg_lambda'],
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
y_pred = model.predict(X_test)

# 모델 평가
r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)