# Linear Discriminant Analysis
# 선형판별분석

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from tensorflow.keras.datasets import cifar100

'''
#1. 데이터
x, y = load_iris(return_X_y=True)       # (150, 4) -> (150, 2)
# [0.9912126 1.       ]

# x, y = load_breast_cancer(return_X_y=True)  
# x, y = load_digits(return_X_y=True) 
# x, y = load_wine(return_X_y=True) 
# x, y = fetch_covtype(return_X_y=True) 

lda = LinearDiscriminantAnalysis()
# n_components는 클래스의 개수 빼기 하나 이하로 가능하다    

x_lda = lda.fit_transform(x, y)
print(x_lda.shape)      # (1797, 9)

lda_EVR = lda.explained_variance_ratio_

cumsum = np.cumsum(lda_EVR)
print(cumsum)
'''
data_list = [load_iris, load_breast_cancer, 
            load_digits, load_wine, fetch_covtype]
# data_list = [load_iris(return_X_y= True), load_breast_cancer(return_X_y= True), 
#             load_digits(return_X_y=True), load_wine(return_X_y= True), fetch_covtype(return_X_y=True)]
data_name = ["load_iris", "load_breast_cancer", "load_digits", "load_wine", "fetch_covtype"]

for i, value in enumerate(data_list):
    print('===============================',data_name[i],'===============================')
    x, y = value(return_X_y=True)
    print('x_shape :', x.shape)
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x, y)
    print('x_lda의 shape :', x_lda.shape)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(cumsum)
print('====================================================')  


# =============================== load_iris ===============================
# x_shape : (150, 4)
# x_lda의 shape : (150, 2)
# [0.9912126 1.       ]
# =============================== load_breast_cancer ===============================
# x_shape : (569, 30)
# x_lda의 shape : (569, 1)
# [1.]
# =============================== load_digits ===============================
# x_shape : (1797, 64)
# x_lda의 shape : (1797, 9)
# [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
#  0.94984789 0.9791736  1.        ]
# =============================== load_wine ===============================
# x_shape : (178, 13)
# x_lda의 shape : (178, 2)
# [0.68747889 1.        ]
# =============================== fetch_covtype ===============================
# x_shape : (581012, 54)
# x_lda의 shape : (581012, 6)
# [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# ====================================================
    
    
    