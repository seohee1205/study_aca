#  전처리 (정규화)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv)
print(train_csv.shape)  # (5497, 13)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

print(test_csv)
print(test_csv.shape)   # (1000, 12)


# 분리형을 수치형으로 바꿔주기
from sklearn.preprocessing import LabelEncoder, RobustScaler          # 데이터 전처리?
le = LabelEncoder()
le.fit(train_csv['type'])   # 0과 1로 인정
aaa = le.transform(train_csv['type'])
# print(aaa)
# print(type(aaa))    # <class 'numpy.ndarray'>
# print(aaa.shape)    # (5497,) 벡터 형태
# print(np.unique(aaa, return_counts= True))      # 몇 개씩 있는지    (array([0, 1]), array([1338, 4159], dtype=int64)) => 0이 1338개, 1이 4159개

train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white']))       # [0 1]     => red = 0, white = 1 
# print(le.transform(['white', 'red']))       # [1 0]     => white = 1, red = 0


# 결측치 제거
print(train_csv.info())     # 결측치 없음

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

# 원핫(다중분류일 때 함)
print('y의 라벨값 : ', np.unique(y))       # [3 4 5 6 7 8 9]
y = to_categorical(y) 

y=np.delete(y, 0, axis=1)       # (5497, 9)
y=np.delete(y, 0, axis=1)
y=np.delete(y, 0, axis=1)       # (5497, 7)

# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 643, stratify= y       # stratify= y :비율대로 너가 뽑아라
)

print(x_train.shape, x_test.shape)  # (4397, 12) (1100, 12)
print(y_train.shape, y_test.shape)  # (4397, 7) (1100, 7)


# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

test_csv = scaler.transform(test_csv)   # test_csv에도 scaler해줘야 함


# #2. 모델 
# model = Sequential()
# model.add(Dense(6, input_dim = 11))
# model.add(Dropout(0.3))
# model.add(Dense(8, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation = 'relu'))
# model.add(Dense(7, activation = 'relu'))
# model.add(Dense(1, activation = 'relu'))
# model.add(Dense(7, activation = 'softmax'))


#(함수형) 모델 구성
input1 = Input(shape=(12,))
dense1 = Dense(132,activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(117,activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(77,activation='relu')(drop2)
drop3 = Dropout(0.4)(dense3)
dense4 = Dense(50,activation='relu')(drop3)
dense5 = Dense(12,activation='relu')(dense4)
dense6 = Dense(5,activation='relu')(dense5)
output1 = Dense(7,activation='softmax')(dense6)

model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics= ['accuracy'])

import datetime
date = datetime.datetime.now()
print(date)  # 2023-03-14 11:11:30.046663
date = date.strftime("%m%d_%H%M") # 시간을 문자로 (월, 일, 시간, 분)
print(date)  # 0314_1116

filepath = './_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'


# 정의하기
es = EarlyStopping(monitor = 'val_accuracy', patience = 1000, mode = 'max',
                   verbose = 1,
                    restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 105,
          validation_split = 0.1,
          verbose = 1,
          callbacks = [es])     # mcp


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis = 1)
y_test = np.argmax(y_test, axis = 1)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis =1)
y_submit += 3


print(y_submit)

# 파일 생성
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

submission['quality'] = y_submit

path_save = './_save/dacon_wine/'

submission.to_csv(path_save + 'submit_0315_0330.csv')


'''

result : [1.1117390394210815, 0.5290908813476562]
acc :  0.5290909090909091

result : [1.1021827459335327, 0.5218181610107422]
acc :  0.5218181818181818

result : [1.0305842161178589, 0.550000011920929]
acc :  0.55

result : [0.9921532273292542, 0.6127272844314575]
acc :  0.6127272727272727

result : [0.9994328022003174, 0.6054545640945435]
acc :  0.6054545454545455

'''
