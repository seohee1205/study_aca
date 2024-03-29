# 가중치는 컴파일 후에 저장해줘야 함
# model.load_weights('./_save/keras26_5_save_weights1.h5')  


# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞히기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용금지

# 1. 삼성전자 28일(화) 종가 맞히기 (점수 배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞히기 (점수 배점 0.7)


#마감시간 : 27일 월 23시 59분 59초        /    28일 화 23시 59분 59초
#윤서희 [삼성 1차] 60,350,07원   (np.round 소수 둘째자리까지)
#윤서희 [삼성]
#첨부파일 : keras53_samsung2_ysh_submit.py       데이터 및 가중치 불러오는 로드가 있어야함
#          keras53_samsung4_ysh_submit.py
#가중치 :  _save/samsung/keras53_samsung2_ysh.h5 / hdf5
#         _save/samsung/keras53_samsung4_ysh.h5 / hdf5



'''
메일 제목:  윤서희 [삼성 1차] 60,350.07원
첨부파일:  keras53_samsung2_ysh_submit.py
가중치:   _save/samsung/keras53_samsung2_ysh.h5 / hdf5
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import concatenate, Concatenate 

#1. 데이터
#1-1. 데이터 불러오기
path = './_data/시험/'
savepath = './_save/samsung/'
mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

datasets1 = pd.read_csv(path + '삼성전자 주가2.csv', index_col=0, encoding='cp949')
datasets2 = pd.read_csv(path + '현대자동차.csv', index_col=0, encoding='cp949')

print(datasets1, datasets2)   # (3260, 16) (3140, 16)

feature_cols = ['시가', '고가', '저가', 'Unnamed: 6', '등락률', '거래량', '기관', '개인', '외국계', '종가']

x1 = datasets1[feature_cols]
x2 = datasets2[feature_cols]

x1 = x1.rename(columns={'Unnamed: 6' : '증감량'})
x2 = x2.rename(columns={'Unnamed: 6' : '증감량'})

y = datasets1['종가']

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)

x1 = x1[:250]
x2 = x2[:250]
y = y[:250]

# 순서 바꾸기
x1 = np.flip(x1, axis = 1)
x2 = np.flip(x2, axis = 1)
y = np.flip(y)


x1 = np.char.replace(x1.astype(str), ',', '').astype(np.float64)
x2 = np.char.replace(x2.astype(str), ',', '').astype(np.float64)
y = np.char.replace(y.astype(str), ',', '').astype(np.float64)

# train, test 분리
x1_train, x1_test, x2_train, x2_test, \
y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, shuffle=False)

# scaler
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)


timesteps=10

def split_x(datasets, timesteps):
    aaa=[]
    for i in range(len(datasets)-timesteps):
        subset = datasets[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x1_train_split = split_x(x1_train, timesteps)
x1_test_split = split_x(x1_test, timesteps)
x2_train_split = split_x(x2_train, timesteps)
x2_test_split = split_x(x2_test, timesteps)

y_train_split = y_train[timesteps:]
y_test_split = y_test[timesteps:]

print(x1_train_split.shape)      # (165, 10, 10)
print(x2_train_split.shape)      #  (165, 10, 10)


# 2-1. 모델1
input1 = Input(shape=(timesteps,10))
conv1d1 =Conv1D(250,2)(input1)
drop1 = Dropout(0.5)(conv1d1)
lstm1 = LSTM(100, activation='swish', name='lstm1')(drop1)
drop2 = Dropout(0.5)(lstm1)
dense1 = Dense(68, activation='swish', name='dense1')(drop2)
dense2 = Dense(64, activation='swish', name='dense2')(dense1)
dense3 = Dense(32, activation='swish', name='dense3')(dense2)
dense4 = Dense(64, activation='swish', name='dense4')(dense3)
output1 = Dense(32, name='output1')(dense4)


# 2-2. 모델2
input2 = Input(shape=(timesteps, 10))
conv1d2 =Conv1D(120,2)(input2)
lstm2 = LSTM(80, activation='swish', return_sequences = True, name='lstm2')(conv1d2)
lstm3 = LSTM(70, activation='swish', name = 'lstm3')(lstm2)
dense11 = Dense(128, activation='swish', name='dense11')(lstm3)
dense12 = Dense(64, activation='swish', name='dense12')(lstm3)
dense13 = Dense(32, activation='swish', name='dense13')(dense12)
dense14 = Dense(34, activation='swish', name='dense14')(dense13)
output2 = Dense(32, name='output2')(dense14)


merge1 = concatenate([output1, output2], name='merge1')
merge2 = Dense(88, activation='swish', name='merge2')(merge1)
merge3 = Dense(64, activation='swish', name='merge3')(merge2)
merge4 = Dense(32, activation='swish', name='merge4')(merge3)
merge5 = Dense(16, activation='swish', name='merge5')(merge4)
merge6 = Dense(8, activation='swish', name='merge6')(merge5)
last_output = Dense(1, name='last')(merge6)

model = Model(inputs=[input1, input2], outputs=[last_output])


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
         verbose = 1, 
         save_best_only= True,
         filepath="".join(['_save/samsung/keras53_samsung2_ysh.h5']))

model.fit([x1_train_split, x2_train_split], 
          y_train_split, 
          epochs = 75, batch_size = 16,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es, mcp])


#4. 평가, 예측
loss = model.evaluate([x1_test_split, x2_test_split], y_test_split)
print('loss : ', loss)

result = model.predict([x1_test_split, x2_test_split])

print('내일의 종가는 바로바로 : ' , result[0])


model.save("./_save/samsung/keras53_samsung2_ysh1.h5")