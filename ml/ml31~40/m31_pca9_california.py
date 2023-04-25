# PCA : 차원(컬럼) 축소(압축)
# target(Y)는 축소 안함 -> X컬럼들만 차원 축소시킴  
# 즉, 타겟값이 없음, 타겟값 생성함(비지도학습 unsupervised learning) : 스케일링 개념
#1. y값을 찾는 비지도학습
#2. 전처리개념 스케일링
# 컬럼 간의 좌표를 찍었을때, 그려지는 직선위로 데이터들의 좌표를 맵핑한다. 

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. DATA
datasets = [
    fetch_california_housing()
]
after = True
for i, v in enumerate(datasets):
    x, y = v.data, v.target
    print(x.shape, y.shape) # (178, 13) (178,)
    if after == True:
        n_pca = x.shape[1] - 1
        # MODEL
        pca = PCA(n_components=n_pca)
        x = pca.fit_transform(x)
        print(n_pca)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=123
        )
        model = RandomForestRegressor(random_state=123, n_jobs=-1)

        # COMPILE
        model.fit(x_train, y_train)

        # PREDICT
        if after == True:
            print('AFTER_PCA', model)
            results = model.score(x_test, y_test)
            print('model_name: ', model)
            print("RESULTS :", results)
    else:
        n_pca = x.shape[1]
        # MODEL
        pca = PCA(n_components=n_pca)
        x = pca.fit_transform(x)
        print(n_pca)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=123
        )
        model = RandomForestRegressor(random_state=123, n_jobs=-1)

        # COMPILE
        model.fit(x_train, y_train)

        # PREDICT
        if after == False:
            print('BEFORE_PCA', model)
            results = model.score(x_test, y_test)
            print('model_name: ', model)
            print("RESULTS :", results)

# model_name:  RandomForestRegressor(n_jobs=-1, random_state=123)
# RESULTS : 0.7786727671384369