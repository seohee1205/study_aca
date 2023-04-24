###################### 실습 ###################
# pca를 통해 0.95 이상인 n_components는 몇 개?
# 0.95 몇 개?
# 0.99 몇 개?
# 0.999 몇 개?
# 1.0 몇 개?
# 힌트 np.argmax

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
(x_train, __ ), (x_test, __ ) = mnist.load_data()     # x만 뽑겠다
# print(__.shape)

# x = np.concatenate((x_train, x_test), axis = 0)     # (70000, 28, 28)
x = np.append(x_train, x_test, axis = 0)        # (70000, 28, 28)
print(x.shape)

# PCA를 적용하기 위해서는 2차원 이상의 배열이어야 함
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)      # (70000, 784)

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)   # 각 주성분이 데이터의 분산을 얼마나 설명할 수 있는지 비율을 계산
print(cumsum)      # 주성분의 누적 설명 분산 비율을 계산

# np.argmax() 함수를 이용하여 누적 설명 분산 비율이 특정 값 이상인 첫 번째 인덱스를 계산
print(np.argmax(cumsum >= 0.95) + 1)     # 154
print(np.argmax(cumsum >= 0.99) + 1)     # 331
print(np.argmax(cumsum >= 0.999) + 1)    # 486
print(np.argmax(cumsum >= 1.0) + 1)      # 713



