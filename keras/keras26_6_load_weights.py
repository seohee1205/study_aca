# 함수형 모델

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model, load_model
from tensorflow.python.keras.layers import Dense,Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 650
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)     # fit의 범위: x_train
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) # 위 두 줄과 같음
x_test = scaler.transform(x_test)

# print(np.min(x_test), np.max(x_test)) # 0.0 1.0


#2. 함수형모델 구성
input1 = Input(shape=(13,)) # 인풋명시, 
dense1 = Dense(10, activation = 'relu')(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(5, activation = 'relu')(dense1)
dense3 = Dense(7, activation = 'relu')(dense2)
dense4 = Dense(2, activation = 'relu')(dense3)
dense5 = Dense(4, activation = 'relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)


# 가중치 로드
# model.load_weights('./_save/keras26_5_save_weights1.h5')  
# 초기 랜덤 값의 웨이트 저장돼 있음
#######################################################################

# 가중치는 컴파일 후에 저장해줘야 함
model.load_weights('./_save/keras26_5_save_weights1.h5')  
# 웨이트는 컴파일 훈련 꼭 해야줘야 함 (웨이트 값만 뽑아오니까!!)



# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train, y_train, epochs = 10)

# 모델 저장
# model.save('./_save/keras26_3_save_model.h5')   # 가중치까지 저장


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)