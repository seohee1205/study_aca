# Linear Discriminant Analysis
# 선형판별분석
# 주어진 데이터를 가장 잘 구분하는 초평면을 찾아내는 방법
# 이 초평면은 클래스 간 분산(between-class variance)과 클래스 내 분산(within-class variance)을
# 최대화, 최소화 하는 방식으로 결정됨
# LDA는 주로 분류 문제에서 사용되며, 입력 변수들이 주어졌을 때 
# 그 입력 변수들이 어떤 클래스에 속하는지 예측하는 모델을 만들 때 사용됨 
# 이를 위해서 LDA는 주어진 데이터 셋에서 클래스 간 분산과 클래스 내 분산을 추정하여 최적의 결정 경계를 찾아냄
# LDA는 이진 분류 뿐 아니라 다중 클래스 분류 문제에서도 사용할 수 있음


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits

#1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)        # (569, 1)
x, y = load_digits(return_X_y=True)                 # (1797, 9)


# pca = PCA(n_components=3)
## pca = PCA()
# x = pca.fit_transform(x)
print(x.shape)      # (150, 64)

# lda = LinearDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis(n_components=3)
# n_components는 클래스의 개수 빼기 하나 이하로 가능하다

x_lda = lda.fit_transform(x, y)
print(x_lda.shape)      # (150, 2)      





