from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(datasets)
print(datasets.feature_names)   # 데이터 열의 이름
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
# 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']

print(datasets.DESCR)   # 데이터셋의 정보
'''
   - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

'''

print(x.shape, y.shape)     # (506, 13) (506,) 스칼라 506개짜리 벡터 1개


# [실습]
# 1. train = 0.7
# 2. R2 = 0.8 이상


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.7, shuffle = True, random_state= 650874)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(14))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 3000, batch_size = 120)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)        

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)