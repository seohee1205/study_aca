from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor, XGBClassifier
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time

#1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 337
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 16),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50),
}

def xgb_hamsu(learning_rate, max_depth, subsample, colsample_bytree,
              reg_lambda, reg_alpha):
    params = {
        'n_estimators': 1000,
        'objective': 'binary:logistic',  # 이진 분류 모델
        # 'eval_metric': 'logloss',  # 평가 지표
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': colsample_bytree,
        'reg_lambda': max(reg_lambda, 0),
        'reg_alpha': reg_alpha,
        'tree_method': 'gpu_hist',  # GPU를 사용하여 학습 (선택 사항)
        'predictor': 'gpu_predictor',  # GPU를 사용하여 예측 (선택 사항)
    }
    
    model = XGBClassifier(**params)

    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds= 50
             )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results

xgb_bo = BayesianOptimization(f=xgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=337
                             )

start_time = time.time()
n_iter = 500
xgb_bo.maximize(init_points=5, n_iter=n_iter)  
end_time = time.time()

print(xgb_bo.max)
print(n_iter, "번 걸린시간 : ", end_time - start_time)

