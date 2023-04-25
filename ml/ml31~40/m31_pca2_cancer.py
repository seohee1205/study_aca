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
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_breast_cancer()
print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)    # (569, 30) (569,)

for i in range(30, 0, -1):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestRegressor(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")


# n_coponets=30,  결과: 0.8699226862679585 
# n_coponets=29,  결과: 0.8695684597393919 
# n_coponets=28,  결과: 0.8718880721683929 
# n_coponets=27,  결과: 0.884206014032743 
# n_coponets=26,  결과: 0.8769538924156365 
# n_coponets=25,  결과: 0.8785917139993318 
# n_coponets=24,  결과: 0.8852496491814233 
# n_coponets=23,  결과: 0.882225392582693 
# n_coponets=22,  결과: 0.8791325760106916 
# n_coponets=21,  결과: 0.8846821249582358 
# n_coponets=20,  결과: 0.8882396257935181 
# n_coponets=19,  결과: 0.8889709321750752 
# n_coponets=18,  결과: 0.8935949214834614 
# n_coponets=17,  결과: 0.8904030738389576 
# n_coponets=16,  결과: 0.8908334781156031 
# n_coponets=15,  결과: 0.8884795856999665 
# n_coponets=14,  결과: 0.8911458068827264 
# n_coponets=13,  결과: 0.8968553291012362 
# n_coponets=12,  결과: 0.8974609421984631 
# n_coponets=11,  결과: 0.8921361176077514 
# n_coponets=10,  결과: 0.8914733711994653 
# n_coponets=9,  결과: 0.8892984964918142 
# n_coponets=8,  결과: 0.8908068159037754 
# n_coponets=7,  결과: 0.8899117273638489 
# n_coponets=6,  결과: 0.8981198797193451 
# n_coponets=5,  결과: 0.9016697627798196 
# n_coponets=4,  결과: 0.9056957567657868 
# n_coponets=3,  결과: 0.8009856331440026 
# n_coponets=2,  결과: 0.8044821917808219 
# n_coponets=1,  결과: 0.6187760775141997 