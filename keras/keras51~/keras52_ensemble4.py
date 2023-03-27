#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)])       # 삼성, 아모레

# 온도, 습도, 강수량
print(x1_datasets.shape)    # (2, 100)

#1-1. 행, 열 바꾸기
x1 = np.transpose(x1_datasets)
print(x1.shape)    # (100, 2)

y1 = np.array(range(2001, 2101))  # 환율
y2 = np.array(range(1001, 1101))  # 금리

#1-2. train, test 분리  ( \: 줄이 너무 길 때 씀, 한 줄이다라는 뜻)
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size = 0.7, random_state= 333 
)
# y_train, y_test = train_test_split(
#     y, train_size = 0.7, random_state= 333 
# )

print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)    # (70,) (30,)
print(y2_train.shape, y2_test.shape)    # (70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape = (2,))
dense1 = Dense(35, activation = 'swish', name = 'stock1')(input1)
dense2 = Dense(24, activation = 'swish', name = 'stock2')(dense1)
dense3 = Dense(12, activation = 'swish', name = 'stock3')(dense2)
output1 = Dense(11, activation = 'swish', name = 'output1')(dense3)

#2-2. 모델 합침(머지)
from tensorflow.keras.layers import concatenate, Concatenate     # 사슬처럼 잇다 / # 소문자: 함수, 대문자: class

#2-5. 분기1 
bungi1 = Dense(40,activation='swish')(output1)
bungi2 = Dense(30,activation='swish')(bungi1)
bungi3 = Dense(20,activation='swish')(bungi2)
bungi4 = Dense(10,activation='swish')(bungi3)
output2 =Dense(1)(bungi4)

#2-6. 분기2
bungi21 = Dense(30,activation='swish')(output1)
bungi22 = Dense(20,activation='swish')(bungi21)
bungi23 = Dense(10,activation='swish')(bungi22)
output3 = Dense(1)(bungi23)

model = Model(inputs = input1, outputs = [output2, output3])

# 큰 모델로 봤을 때, input과 output만 맞게 하면 됨 (중간 모델의 아웃풋은 노상관~)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
import time
start = time.time()

model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit(x1_train, [y1_train, y2_train], epochs = 2000,
          batch_size = 6, validation_split = 0.2, callbacks = [es])

end = time.time()

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_squared_error

results = model.evaluate(x1_test,
                      [y1_test, y2_test])
print('results : ', results)

y_predict = model.predict(x1_test)
# np.array(y_predict = model.predict([x1_test, x2_test, x3_test]))  # np.array로 하면 shape로 볼 수 있음

print(y_predict)
# 리스트는 파이썬 기본 자료형이기 때문에 shape 함수를 사용할 수 없음, 따라서 len을 사용하여 데이터를 길이로 확인해야함
print(len(y_predict), len(y_predict[0]))    # 2, 30 / y가 몇 개인지, 0번 째 몇 개인지 

r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])
print('r2 스코어 : ', (r2_1+r2_2) / 2)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse1 = RMSE(y1_test, y_predict[0])              # RMSE 함수 사용
rmse2 = RMSE(y2_test, y_predict[1])              # RMSE 함수 사용
print("RMSE : ", (rmse1 + rmse2) / 2)

print('걸린 시간 : ', np.round(end-start, 2))


