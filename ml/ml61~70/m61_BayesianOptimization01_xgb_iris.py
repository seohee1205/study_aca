# [실습] 맹그러# BayesianOptimization : 최댓값 찾기 [함수(최댓값 뽑는 함수정의), 파라미터의 범위 준비] 
# 회귀 평가지표 : mse, mae(최솟값이므로 -넣기) or r2(최댓값)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')
# *UserWarning: 'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. 
# Pass 'early_stopping()' callback via 'callbacks' argument instead.

#1. 데이터 
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

bayesian_params = {
    'learning_rate' : (0.01, 1),
    'max_depth' : (3,16),
    'gamma' : (0,10),        
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),          #subsample 범위 : 0~1사이여야함  min,max / dropout과 비슷한 개념 (훈련을 시킬때의 샘플 양)
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0,1),      #0~1
    'colsample_bynode' : (0, 1),
    'reg_lambda' : (-0.001, 10),      #reg_lambda : 무조건 양수만     max
    'reg_alpha' : (0.01, 50)
}

###파라미터 범위 지정시 주의할 점###
#1. LightGBMError: Parameter num_leaves should be of type int, got "37.582780271475926"
#   =>param형태가 실수로 들어감 따라서, 실수를 정수로 바꿔주고, 반올림해주겠다!
#2. 파라미터의 범위를 벗어나서는 안됨 
#   => 모델 정의에서 쓸수있는 범위로 변환 가능하지만 최대한 파라미터범위내에서 잡아주는 것이 더 좋음 (파라미터 범위에 대한 이해)

#모델 정의 
def lgbm_hamsu(learning_rate, max_depth, gamma,min_child_weight,subsample,colsample_bytree,colsample_bylevel,colsample_bynode,reg_lambda,reg_alpha):
    params = { 
        'n_estimators' : 20,
        'learning_rate' : learning_rate,   
        'max_depth' : int(round(max_depth)),
        'gamma' : gamma,
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1), 0),     #무조건 0~1사이 
        'colsample_bytree' : colsample_bytree,  
        'colsample_bylevel' : colsample_bylevel,  
        'colsample_bynode' : colsample_bynode,  
        'reg_lambda' : max(reg_lambda, 0),          #무조건 양수만  (위의 범위에서 -0.01이 선택되어 들어오더라도 여기서 쓸수있는 범위로 변환 '0'으로 바뀌어서 들어감) 
        'reg_alpha' : reg_alpha                                       #-최대한 위에서 파라미터 범위내로 잡아주는게 좋음 
        }
    
    model = XGBClassifier(**params)
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results


#BayesianOptimization 정의
lgbm_bo = BayesianOptimization(f = lgbm_hamsu, 
                               pbounds= bayesian_params,
                               random_state=337
                               )


start_time = time.time()
n_iter = 100
lgbm_bo.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(lgbm_bo.max)
print(n_iter, "번 걸린시간:", end_time-start_time)


# {'target': 0.9666666666666667, 'params': {'colsample_bylevel': 0.4856854869079368, 
# 'colsample_bynode': 0.17923221644163645, 'colsample_bytree': 0.5485108817982015, 
# 'gamma': 0.0, 'learning_rate': 0.20160857225003212, 'max_depth': 11.515389662339736, 
# 'min_child_weight': 6.584722927365092, 'reg_alpha': 3.629438919024237, 
# 'reg_lambda': 5.233192248011521, 'subsample': 1.0}}
# 100 번 걸린시간: 25.77003765106201