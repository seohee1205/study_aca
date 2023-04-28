import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (178, 13) (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.Series(y).value_counts().sort_index())     # sort_index: 순서대로
# 0    59
# 1    71
# 2    48
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
x = x[:-25]
y = y[:-25]
print(x.shape, y.shape)     # (153, 13) (153,)
print(y)
print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    23

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.75, shuffle=True, random_state= 321,
    stratify= y
)
# print(pd.Series(y_train).value_counts().sort_index())
# 0    44
# 1    53
# 2    17

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=377)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', score)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro)', f1_score(y_test, y_predict, average= 'macro'))
# print('f1_score(micro)', f1_score(y_test, y_predict, average= 'micro'))


# model.score :  0.9487179487179487
# accuracy_score :  0.9487179487179487
# f1_score(macro) 0.9439984430496765
# f1_score(micro) 0.9487179487179487

# f1_score: 이진 분류에서 인지 아닌지 맞히는, 높으면 장땡~!
# 한개의 클래스가 너무 작거나 , 많거나 할 때 acc보다 더 정확한 지표

print("============== SMOTE 적용 후 =============")
smote = SMOTE(random_state= 321, k_neighbors= 8) # 디폴트 5
x_train, y_train = smote.fit_resample(x_train.copy(), y_train.copy())
# print(x_train.shape, y_train.shape)     # (159, 13) (159,)
# print(pd.Series(y_train).value_counts().sort_index())
# 0    53
# 1    53
# 2    53

#2-2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=321)

#3-2. 훈련
model.fit(x_train, y_train)

#4-2. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', score)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro)', f1_score(y_test, y_predict, average= 'macro'))





