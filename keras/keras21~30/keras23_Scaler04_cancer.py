#  전처리 (정규화)
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))      # <class 'numpy.ndarray'>
print(x)


# 보스턴 임포트 못 하는 사람 (1.2 부터 안 되니까 1.1 버전 설치)
# pip uninstall scikit-learn     # 사이킷런 삭제
# pip install scikit- learn==1.1.0      # 1.1 버전 설치

# print(np.min(x), np.max(x)) # 0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)     # 변환
# print(np.min(x), np.max(x)) # 0.0 1.0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 555
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0
 
# 정규화: 0 ~ 1 사이로 만들어주는 것,  y 제외 x의 훈련 데이터만
# => 정규화 했을 때 무조건 성능 좋아지는 거 X, 좋아질 수도 있고 안 좋아질 수 있음
# 정규화 비율 구하는 공식
# X / Max	->	X - min / max - min
# ex)	10~100 일 때, X - 10 / 100 - 10

#2. 모델 
model = Sequential()
model.add(Dense(1, input_dim = 30))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics= ['accuracy', 'acc', 'mse'], #'mean_suquared_error'] # 결과에 mse, mae도 보고 싶으면 metrics에 추가하면 됨.
              ) # 'accuracy' = 'acc'

# 정의하기 (얼리스탑핑)
es = EarlyStopping(monitor = 'val_loss', patience = 25, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)

hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 25,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('암')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()

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





'''
# scaler = MinMaxScaler() 
loss :  [0.24332603812217712, 0.9210526347160339, 0.9210526347160339, 0.19158348441123962]
r2 스코어 :  0.5714079292967784
RMSE :  0.4377024971283921

# scaler = StandardScaler() 
loss :  [0.32318115234375, 0.9473684430122375, 0.9473684430122375, 0.6583656072616577]
r2 스코어 :  0.5052044114044145
RMSE :  0.8113973130187211

# scaler = MaxAbsScaler()
loss :  [9.336153030395508, 0.3947368562221527, 0.3947368562221527, 0.6257907748222351]
r2 스코어 :  -840.7065343597113
RMSE :  0.7910693651608931

# scaler = RobustScaler()
loss :  [9.336153030395508, 0.3947368562221527, 0.3947368562221527, 0.6080530881881714]
r2 스코어 :  -91821.97791351091
RMSE :  0.7797775818975043

'''