import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터셋
kaggle_bike_path = './_data/kaggle_bike/'
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col=0).dropna()

# 데이터를 살펴보고 시각화
print(kaggle_bike.head())

# 박스 플롯
# ddarung.plot.box()
# plt.show()

# 정보 및 통계 요약
# ddarung.info()
# print(ddarung.describe())

# 히스토그램
kaggle_bike.hist(bins=50)
plt.show()

# 데이터 분할
# 가정: 'target'이라는 열이 목표 변수임
y = kaggle_bike[['casual', 'registered', 'count']]
x = kaggle_bike.drop(['count'], axis=1)

x[['holiday', 'workingday']] = np.log1p(x[['holiday', 'workingday']])

y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1234)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# 모델 생성
model = RandomForestRegressor(random_state=1234)

# 모델 훈련
model.fit(x_train, y_train_log)

# 모델 평가
score = model.score(x_test, y_test_log)
print('score:', score)

# R2 스코어
print('R2 score:', r2_score(y_test, model.predict(x_test)))

# score: 0.9996372999330426
# R2 score: -2.6372863481269433
