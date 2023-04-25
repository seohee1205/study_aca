import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target']  
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle =True, random_state= 650874, train_size= 0.7
)

#2. 모델 구성
model = Sequential()
model.add(Dense(15, input_dim = 10))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 20, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,   
              restore_best_weights= True)  


hist = model.fit(x_train, y_train, epochs = 1000, batch_size =25,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )

# print("=============================================")
# print(hist)
# # <tensorflow.python.keras.callbacks.History object at 0x00000227C99A01F0>
# print("=============================================")
# print(hist.history)
# print("=============================================")
# print(hist.history['loss'])
# print("======================발로스=======================")
# print(hist.history['val_loss'])
# print("======================발로스=======================")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('디아벳')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()

# val_loss가 loss보다 높은 위치에 있음

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)



# loss :  2645.674560546875
# r2 스코어 :  0.022894538003540688
# RMSE :  51.43611939843769