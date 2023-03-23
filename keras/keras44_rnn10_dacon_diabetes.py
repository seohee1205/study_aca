from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense,GRU, Input, Dropout, MaxPooling2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pandas as pd
import time

#1. 데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv.shape)  # (652, 9)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

# 결측치
print(train_csv.info())     # 결측치 없음

x = train_csv.drop(['Outcome'], axis = 1)
print(x)
y = train_csv['Outcome']
print(y)

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 789
)

print(x_train.shape, y_train.shape)  # (521, 8) (521,)
print(x_test.shape, y_test.shape)   # (131, 8) (131,)


# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2차원을 4차원으로 만들어주기
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
test_csv = test_csv.reshape(-1, 8, 1)


#2. 모델 구성
input1 = Input(shape=(8, 1))
GRU1 = GRU(54, activation='relu',  return_sequences = True)(input1)
GRU2 = GRU(34, activation='relu',  return_sequences = True)(GRU1)
GRU3 = GRU(24)(GRU2)
dense1 = Dense(16,activation='relu')(GRU3)
dense2 = Dense(12,activation='relu')(dense1)
output1 = Dense(1)(dense2)
model = Model(inputs=input1, outputs=output1)
# model.summary()


#3. 컴파일, 훈련
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 30,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

#5 파일 저장
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

path_save = './_save/dacon_diabetes/'
submission.to_csv(path_save + 'submit_0323_1215.csv')
print('걸린 시간 : ', np.round(end-start, 2))


