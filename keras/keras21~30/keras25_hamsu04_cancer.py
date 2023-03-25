#  전처리 (정규화)
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x)


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
 

#2. 모델 
# model = Sequential()
# model.add(Dense(1, input_dim = 30))
# model.add(Dense(5, activation = 'relu'))
# model.add(Dense(7, activation = 'relu'))
# model.add(Dense(12, activation = 'relu'))
# model.add(Dense(4, activation = 'relu'))
# model.add(Dense(3, activation = 'relu'))
# model.add(Dense(1))

#(함수형) 모델 구성
input1 = Input(shape=(30,)) # 인풋명시, 
dense1 = Dense(1)(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(5, activation = 'relu')(dense1)
dense3 = Dense(7, activation = 'relu')(dense2)
dense4 = Dense(12, activation = 'relu')(dense3)
dense5 = Dense(4, activation = 'relu')(dense4)
dense6 = Dense(3, activation = 'relu')(dense5)
output1 = Dense(1, activation= 'sigmoid')(dense6)
model = Model(inputs = input1, outputs = output1)


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
loss :  [0.40389567613601685, 0.9122806787490845, 0.9122806787490845, 0.28770172595977783]
r2 스코어 :  0.6264073447593612
RMSE :  0.5363783206973398
 
# scaler = StandardScaler() 
loss :  [9.336153030395508, 0.3947368562221527, 0.3947368562221527, 0.67178875207901]
r2 스코어 :  -70.52871231138708
RMSE :  0.8196271648619659

# scaler = MaxAbsScaler()
loss :  [0.3525400459766388, 0.9385964870452881, 0.9385964870452881, 0.7037107944488525]
r2 스코어 :  0.2264552271378436
RMSE :  0.838874703440049

# scaler = RobustScaler()
loss :  [9.336153030395508, 0.3947368562221527, 0.3947368562221527, 1.192171335220337]
r2 스코어 :  -8.422600434579955
RMSE :  1.0918659462584173


'''

