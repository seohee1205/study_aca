# [실습]

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
data_list = [load_diabetes, fetch_california_housing]
data_name_list = ['디아벳', '캘리포니아']

model_list = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
model_name_list = ['엑스지비', '엘지비엠', '캣부스트']

xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

list =[]
for i, v in enumerate(data_list):
    x, y = v(return_X_y= True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state=123, shuffle=True
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    for j, v2 in enumerate(model_list):
        
        model = VotingRegressor(
            estimators=[('XG', xg), ('LG', lg), ('CAT', cat)],
        )
        regressor = [xg, lg, cat]
        for model2 in regressor:
            model2.fit(x_train, y_train)
            y_predict = model2.predict(x_test)
            score2 = r2_score(y_test, y_predict)
            class_name = model2.__class__.__name__
            print('==========', data_name_list[i], '==========')
            print("{0} r2 : {1:.4f}".format(class_name, score2))
            # print('model.score : ', model.score(x_test, y_test))
            print('voting r2 : ', r2_score(y_test, y_predict))
            list.append(score2)



# ========== 디아벳 ==========
# XGBRegressor r2 : 0.4607
# voting r2 :  0.4606593163373558
# ========== 디아벳 ==========
# LGBMRegressor r2 : 0.5256
# voting r2 :  0.5256028696915696
# ========== 디아벳 ==========
# CatBoostRegressor r2 : 0.5383
# voting r2 :  0.5382592137285847
# ========== 디아벳 ==========
# XGBRegressor r2 : 0.4607
# voting r2 :  0.4606593163373558
# ========== 디아벳 ==========
# LGBMRegressor r2 : 0.5256
# voting r2 :  0.5256028696915696
# ========== 디아벳 ==========
# CatBoostRegressor r2 : 0.5383
# voting r2 :  0.5382592137285847
# ========== 디아벳 ==========
# XGBRegressor r2 : 0.4607
# voting r2 :  0.4606593163373558
# ========== 디아벳 ==========
# LGBMRegressor r2 : 0.5256
# voting r2 :  0.5256028696915696
# ========== 디아벳 ==========
# CatBoostRegressor r2 : 0.5383
# voting r2 :  0.5382592137285847
# ========== 캘리포니아 ==========
# XGBRegressor r2 : 0.8331
# voting r2 :  0.8331072804876352
# ========== 캘리포니아 ==========
# LGBMRegressor r2 : 0.8413
# voting r2 :  0.8412620449195014
# ========== 캘리포니아 ==========
# CatBoostRegressor r2 : 0.8571
# voting r2 :  0.8570785719504063
# ========== 캘리포니아 ==========
# XGBRegressor r2 : 0.8331
# voting r2 :  0.8331072804876352
# ========== 캘리포니아 ==========
# LGBMRegressor r2 : 0.8413
# voting r2 :  0.8412620449195014
# ========== 캘리포니아 ==========
# CatBoostRegressor r2 : 0.8571
# voting r2 :  0.8570785719504063
# ========== 캘리포니아 ==========
# XGBRegressor r2 : 0.8331
# voting r2 :  0.8331072804876352
# ========== 캘리포니아 ==========
# LGBMRegressor r2 : 0.8413
# voting r2 :  0.8412620449195014
# ========== 캘리포니아 ==========
# CatBoostRegressor r2 : 0.8571
# voting r2 :  0.8570785719504063

