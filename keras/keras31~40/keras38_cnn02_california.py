from sklearn.datasets import fetch_california_housing
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

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

########################
# print(type(x))      # <class 'numpy.ndarray'>
# print(x.shape, y.shape)     # (20640, 8) (20640,)
# print(datasets.info())
# print(datasets.DESCR)
# print(datasets.describe)
# print(datasets.columns)
# print(datasets.feature_names)
#########################

#1-2. x, y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 333
)

print(x_train.shape, y_train.shape) # (16512, 8) (16512,)
print(x_test.shape, y_test.shape)   # (4128, 8) (4128,)

#1-3.  2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(16512, 8*1*1 )
x_test = x_test.reshape(4128, 8*1*1)


#1-3 scaler
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(16512, 8, 1, 1)
x_test = x_test.reshape(4128, 8, 1, 1)

#2. 모델
input1 = Input(shape=(8, 1, 1))
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

model.fit(x_train, y_train, epochs = 1000, batch_size = 500,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# result :  0.3739278316497803
# r2 스코어 :  0.7065109318931789