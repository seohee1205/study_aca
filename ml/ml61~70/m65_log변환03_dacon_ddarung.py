import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터셋
ddarung_path = './_data/dacon_ddarung/'
ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col=0).dropna()

# 데이터를 살펴보고 시각화
print(ddarung.head())

# 박스 플롯
# ddarung.plot.box()
# plt.show()

# 정보 및 통계 요약
# ddarung.info()
# print(ddarung.describe())

# 히스토그램
ddarung.hist(bins=50)
plt.show()

# 데이터 분할
# 가정: 'target'이라는 열이 목표 변수임
y = ddarung[['hour_bef_visibility', 'hour_bef_precipitation']]
x = ddarung.drop(['count'], axis=1)

x[['hour', 'hour_bef_precipitation']] = np.log1p(x[['hour', 'hour_bef_precipitation']])

y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=1234)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#2. 모델
model = RandomForestRegressor(random_state=1234)

#3. 훈련
model.fit(x_train, y_train_log)

#4. 평가, 예측
score = model.score(x_test, y_test_log)
print('score:', score)

# R2 스코어
print('R2 score:', r2_score(y_test, model.predict(x_test)))

# print("로그 -> 지수 r2 : ", r2_score(y_test, np.expm1(model.predict(x_test))))

# score: 0.999502044859035
# R2 score: -37.99997660127003

