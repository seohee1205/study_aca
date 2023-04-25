# Linear Discriminant Analysis
# 선형판별분석
# 상민이가 회귀에서 된다고 했다
# 성호는 y에 라운드 때렸어

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from tensorflow.keras.datasets import cifar100
from sklearn.datasets import load_diabetes, fetch_california_housing


#1. 데이터
x, y = load_diabetes(return_X_y=True)                # (442, 10)
# x, y = fetch_california_housing(return_X_y=True)   # 실수형 데이터라 안 되는데, round처리해서 정수형으로 바뀌면서 class로 판단함
# y = np.round(y)
print(y)
print(len(np.unique(y)))     # 214

# lda = LinearDiscriminantAnalysis(n_components = 101)
lda = LinearDiscriminantAnalysis()
# n_components는 클래스의 개수 빼기 하나 이하로 가능하다    

x_lda = lda.fit_transform(x, y)
print(x_lda.shape)      # (20640, 5)

### 회귀는 원래 안 돼. 하지만 diabetes는 정수형이라서 LDA에서 y의 클래스로 잘못 인식한 거야
# 그래서 돌아간 거여

# 성호는 캘리포니아에서 라운드 처리했음
# 정수형으로 바뀌면서 클래스로 인식돼서 돌아간 거야

# 회귀데이터는 원칙적으로 에러인데
# 위처럼 돌리고 싶으면 돌려도 돼
# 성능 보장 못 함 ㅋㅋ
