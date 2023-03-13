# 이미지 
#  전처리 (정규화)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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


# 보스턴 임포트 못 하는 사람 (1.2 부터 안 되니까 1.1 버전 설치)
# pip uninstall scikit-learn     # 사이킷런 삭제
# pip install scikit- learn==1.1.0      # 1.1 버전 설치

# print(np.min(x), np.max(x)) # 0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)     # 변환
# print(np.min(x), np.max(x)) # 0.0 1.0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 650
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0



# 정규화: 0 ~ 1 사이로 만들어주는 것,  y 제외 x의 훈련 데이터만
# => 정규화 했을 때 무조건 성능 좋아지는 거 X, 좋아질 수도 있고 안 좋아질 수 있음
# 정규화 비율 구하는 공식
# X / Max	->	X - min / max - min
# ex)	10~100 일 때, X - 10 / 100 - 10

#2. 모델 구성
model = Sequential()
# model.add(Dense(1, input_dim = 13))
model.add(Dense(1, input_shape = (13,)))    # 열의 개수를 벡터 형식을 명시해줌 => (열, )

# if 데이터가 3차원이면(시계열 데이터)
# (1000, 100, 1) ---> input_shape=(100, 1)
# if 데이터가 4차원이면(이미지 데이터)
# (60000, 32, 32, 3) ---> input_shape=(32, 32, 3)   # 행무시



#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 25, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    # val-loss를 기준으로 할 것이고, 5번 참을 것이다. / val-loss의 최솟값을 찾아라
              restore_best_weights= True)  # 최적(최소 loss)의 w 값을 반환한다.


hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 30,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )


matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# loss :  20.41268539428711

