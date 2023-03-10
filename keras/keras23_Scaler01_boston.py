#  전처리 (정규화)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

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
    x, y, train_size= 0.8, random_state= 333
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  
# = scaler = StandardScaler()   # 표준 정규표를 만들 때
# = scaler = MaxAbsScaler()
# = scaler = RobbustScaler()
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
model.add(Dense(1, input_dim = 13))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
