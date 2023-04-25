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
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
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
'''
# [실습]

data_name_list = [load_iris(return_X_y= True), load_breast_cancer(return_X_y= True), 
            load_digits(return_X_y=True), load_wine(return_X_y= True), fetch_covtype(return_X_y=True)]

data_name = ["load_iris", "load_breast_cancer", "load_digits", "load_wine", "fetch_covtype"]


for i, v in enumerate(data_name_list):
    x, y = v
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 337)

    print('=================', '변환 전', data_name[i], '===============')
    # 모델 구성
    model = RandomForestClassifier(random_state=337)
    # 컴파일, 훈련
    model.fit(x_train, y_train)
    # 평가, 예측
    result = model.score(x, y)
    print("acc :", result)

    print('=================', '변환 후', data_name[i], '===============')
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x, y)
    
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.8, random_state= 337)
    print(data_name[i], ':', x.shape, '->', x_lda.shape)
    
    # 모델 구성
    model = RandomForestClassifier(random_state=337)
    # 컴파일, 훈련 
    model.fit(x_train1, y_train1)
    # 평가, 예측
    result1 = model.score(x_test1, y_test1)
    print("acc :", result1)
    

# ================= 변환 전 load_iris ===============
# acc : 0.9933333333333333
# ================= 변환 후 load_iris ===============
# load_iris : (150, 4) -> (150, 2)
# acc : 0.9666666666666667
# ================= 변환 전 load_breast_cancer ===============
# acc : 0.9912126537785588
# ================= 변환 후 load_breast_cancer ===============
# load_breast_cancer : (569, 30) -> (569, 1)
# acc : 0.956140350877193
# ================= 변환 전 load_digits ===============
# acc : 0.993322203672788
# ================= 변환 후 load_digits ===============
# load_digits : (1797, 64) -> (1797, 9)
# acc : 0.9666666666666667
# ================= 변환 전 load_wine ===============
# acc : 0.9887640449438202
# ================= 변환 후 load_wine ===============
# load_wine : (178, 13) -> (178, 2)
# acc : 0.9444444444444444
# ================= 변환 전 fetch_covtype ===============
# acc : 0.9910707524113099
# ================= 변환 후 fetch_covtype ===============
# fetch_covtype : (581012, 54) -> (581012, 6)
# acc : 0.9553539925819471
