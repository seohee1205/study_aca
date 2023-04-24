# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.

# [실습]
# for문 써서 한번에 돌려
# 기본결과 : 0.23131244
# 차원 1개 축소: 0.3341432
# 차원 2개 축소: 0.423414
# ...

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. DATA
datasets = [
    fetch_covtype()
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

