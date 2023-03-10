# https://dacon.io/edu/1009

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv)
print(train_csv.shape)  # (652, 9)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

print(test_csv)
print(test_csv.shape)   # (116, 8)

#====================================
print(train_csv.columns)
print(test_csv.columns)
print(train_csv.describe())
print(type(train_csv))

########## 결측치 처리 ###########
# 결측치 제거
print(train_csv.info())  # 결측치 없음

######## train_csv 데이터에서 x와 y 분리

x = train_csv.drop(['Outcome'], axis = 1)
print(x)
y = train_csv['Outcome']
print(y)

############# x와 y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, train_size= 0.9, random_state= 3040447, stratify = y
)

print(x_train.shape, x_test.shape)  # (521, 8) (131, 8)
print(y_train.shape, y_test.shape)  # (521,) (131,)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 8))
model.add(Dense(9, activation = 'linear'))
model.add(Dense(9, activation = 'linear'))
model.add(Dense(7, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics= ['accuracy', 'mse'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)

hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 10,
          validation_split = 0.07,
          verbose = 1,
          callbacks = [es])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('당뇨병')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)
y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# 파일 생성
y_submit =  np.round(model.predict(test_csv))
# print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
# print(submission)
submission['Outcome'] = y_submit
# print(submission)

path_save = './_save/dacon_diabetes/'
submission.to_csv(path_save + 'submit_0309_0445.csv')


# acc :  0.7727272727272727
