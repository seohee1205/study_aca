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
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_wine()
print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
# 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
# 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)    # (178, 13) (178,)

for i in range(13, 0, -1):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestRegressor(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")


# n_coponets=13,  결과: 0.7326769230769231 
# n_coponets=12,  결과: 0.732646153846154 
# n_coponets=11,  결과: 0.7413142857142858 
# n_coponets=10,  결과: 0.7495780219780219 
# n_coponets=9,  결과: 0.7645582417582417 
# n_coponets=8,  결과: 0.7571076923076923 
# n_coponets=7,  결과: 0.7634065934065934 
# n_coponets=6,  결과: 0.7591384615384615 
# n_coponets=5,  결과: 0.7389274725274726 
# n_coponets=4,  결과: 0.7709230769230769 
# n_coponets=3,  결과: 0.6609362637362637 
# n_coponets=2,  결과: 0.4280527472527472 
# n_coponets=1,  결과: 0.2537318681318683 
