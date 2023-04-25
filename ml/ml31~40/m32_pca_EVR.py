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
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target
# print(x.shape, y.shape)     # (569, 30) (569,)


for i in range(10):
    pca = PCA(n_components= 10-i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,shuffle=True,)
    #2. 모델
    model = RandomForestRegressor(random_state=123)
    #3. 훈련
    model.fit(x_train, y_train)
    #4. 평가, 예측
    results = model.score(x_test, y_test)
    print("결과 : ", results)
 
 
   
# 결과 :  0.8914733711994653
# 결과 :  0.8892984964918142
# 결과 :  0.8908068159037754
# 결과 :  0.8899117273638489
# 결과 :  0.8981198797193451
# 결과 :  0.9016697627798196
# 결과 :  0.9056957567657868
# 결과 :  0.8009856331440026
# 결과 :  0.8044821917808219
# 결과 :  0.6187760775141997
    
    

