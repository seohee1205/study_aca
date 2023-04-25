# Linear Discriminant Analysis
# 선형판별분석
# 컬럼의 개수가 클래스의 개수보다 작을 때
# 디폴트로 돌아가나?

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from tensorflow.keras.datasets import cifar100


#1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)        # (569, 1)
x, y = load_digits(return_X_y=True)                 # (1797, 9)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)    # (50000, 32, 32, 3)

x_train = x_train.reshape(50000, 32*32*3)



pca = PCA(n_components=98)
## pca = PCA()
x_train = pca.fit_transform(x_train)


# lda = LinearDiscriminantAnalysis(n_components = 101)
lda = LinearDiscriminantAnalysis()  # # 클래스의 위치 표시 / 디폴트 => 클래스 -1 or n_feature에서 최소값이 나옴 ( 즉 여기선 디폴트 99보다 줄여준 98이 더 작으니까 98로 나옴)

x = lda.fit_transform(x_train,y_train) 
print(x.shape) #(50000, 98)

# n_components는 클래스의 개수 빼기 하나 이하로 가능하다    

x_lda = lda.fit_transform(x_train, y_train)
print(x_lda.shape)      # (50000, 98)       or 




# # 지피티가 그려줌
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris

# # iris 데이터셋 로드
# iris = load_iris()

# # 데이터셋에서 꽃잎의 길이와 폭 정보 추출
# X = iris.data[:, 2:]    # petal length, petal width
# y = iris.target

# # scatter plot 그리기
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.title('iris scatter ')