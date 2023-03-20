from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Input, Dropout, MaxPooling2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pandas as pd

#1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv.shape)  # (5497, 13)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)


# 결측치
print(train_csv.isnull().sum())     # 결측치 없음

#1-1. x, y 데이터 분리
x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

# 원핫(다중분류일 때 함)
print('y의 라벨값 : ', np.unique(y))       # [3 4 5 6 7 8 9]
y = to_categorical(y) 

y=np.delete(y, 0, axis=1)      
y=np.delete(y, 0, axis=1)
y=np.delete(y, 0, axis=1)


# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 777
)

print(x_train.shape, y_train.shape)     # (4397, 12) (4397,)
print(x_test.shape, y_test.shape)       # (1100, 12) (1100,)


# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = x_train.reshape(-1, 12, 1, 1)
x_test = x_test.reshape(-1, 12, 1, 1)
test_csv = test_csv.reshape(-1, 12, 1, 1)

#2. 모델
input1 = Input(shape=(12, 1, 1))
conv1 = Conv2D(64, (3,3),
               padding='same', 
               activation='relu')(input1)
conv2 = Conv2D(54, (3,3),
               padding='same', 
               activation='relu')(conv1)
# mp1 = MaxPooling2D()
# pooling1 = mp1(conv2)
conv3 = Conv2D(64, (3,3),
               padding='same', 
               activation='relu')(conv2)
# pooling2 = mp1(conv3)
flat1 = Flatten()(conv3)
dense1 = Dense(26,activation='relu')(flat1)
dense2 = Dense(16,activation='relu')(dense1)
dense3 = Dense(12,activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 50,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# 파일 생성
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['quality'] = y_submit
path_save = './_save/dacon_wine/'
submission.to_csv(path_save + 'submit_0320_0805.csv')

