# Linear Discriminant Analysis
# 선형판별분석

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
# x, y = load_breast_cancer(return_X_y=True)  
x, y = load_digits(return_X_y=True)          

lda = LinearDiscriminantAnalysis()
# n_components는 클래스의 개수 빼기 하나 이하로 가능하다    

x_lda = lda.fit_transform(x, y)
print(x_lda.shape)      # (50000, 98)       or 

