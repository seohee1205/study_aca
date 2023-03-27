# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞히기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라
# 앙상블 사용하기
# 제공된 데이터 외 추가 데이터 사용 금지

#1. 삼성전자 28일(화) 종가 맞히기 (점수배점 0.3)
#2. 삼성전자 29일(수) 아침 시가 맞히기 (점수배점 0.7)
'''
마감시간: 27일(월) 23시 59분 59초 / 28일(화) 23시 59분 59초

메일 제목:  윤서희 [삼성 1차] 60,350.07원
           윤서희 [삼성 2차] 60,350.07원
첨부파일:  keras53_samsung2_ysh_submit.py
          keras54_samsung4_ysh_submit.py
가중치:   _save/samsung/keras53_samsung2_ysh.h5 / hdf5
         _save/samsung/keras53_samsung4_ysh.h5 / hdf5
'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
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

datasets_samsung = pd.read_csv(path + '삼성전자 주가2.csv', index_col=0, encoding='cp949')
datasets_hyundai = pd.read_csv(path + '현대자동차.csv', index_col=0, encoding='cp949')

print(datasets_samsung.shape, datasets_hyundai.shape)   # (3260, 16) (3140, 16)
# print(datasets_samsung.columns, datasets_hyundai.columns)
# print(datasets_samsung.info(), datasets_hyundai.info())
# print(datasets_samsung.describe(), datasets_hyundai.describe())
# print(type(datasets_samsung), type(datasets_hyundai))

samsung_x = np.array(datasets_samsung.drop(['전일비', '종가'], axis=1))
samsung_y = np.array(datasets_samsung['종가'])
hyundai_x = np.array(datasets_hyundai.drop(['전일비', '종가'], axis=1))

samsung_x = samsung_x[:1207, :]
samsung_y = samsung_y[:1207]
hyundai_x = hyundai_x[:1207, :]

print(samsung_x.shape, samsung_y.shape)     # (1207, 14) (1207,)
print(hyundai_x.shape)     # (1207, 14) (1207,)

samsung_x = np.char.replace(samsung_x.astype(str), ',', '').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',', '').astype(np.float64)
hyundai_x = np.char.replace(hyundai_x.astype(str), ',', '').astype(np.float64)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, \
hyundai_x_train, hyundai_x_test = train_test_split(
    samsung_x, samsung_y, hyundai_x, train_size=0.7, shuffle=False)

# scaler
scaler = MinMaxScaler()
samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test = scaler.transform(samsung_x_test)
hyundai_x_train = scaler.transform(hyundai_x_train)
hyundai_x_test = scaler.transform(hyundai_x_test)

#
timesteps = 20
def split_x(dataset, timesteps):
    aaa = []    # aaa라는 빈 리스트를 만들어라
    for i in range(len(dataset) - timesteps):        
        subset = dataset[i : (i + timesteps)]               
        aaa.append(subset)                                  
    return np.array(aaa)    

samsung_x_train_split = split_x(samsung_x_train, timesteps)
samsung_x_test_split = split_x(samsung_x_test, timesteps)
hyundai_x_train_split = split_x(hyundai_x_train, timesteps)
hyundai_x_test_split = split_x(hyundai_x_test, timesteps)

samsung_y_train_split = samsung_y_train[timesteps:]
samsung_y_test_split = samsung_y_test[timesteps:]


print(samsung_x_train_split.shape)      # (820, 20, 14)
print(hyundai_x_train_split.shape)      # (820, 20, 14)


#2-1. 모델1
input1 = Input(shape = (20, 14))
dense1 = LSTM(35, activation = 'swish', name = 'stock1')(input1)
dense2 = Dense(24, activation = 'swish', name = 'stock2')(dense1)
dense3 = Dense(12, activation = 'swish', name = 'stock3')(dense2)
output1 = Dense(11, activation = 'swish', name = 'output1')(dense3)

#2-2. 모델 2
input2 = Input(shape = (20, 14))
dense11 = LSTM(30, name = 'weather1')(input2)
dense12 = Dense(16, activation = 'swish', name = 'weather2')(dense11)
dense13 = Dense(52, activation = 'swish', name = 'weather3')(dense12)
dense14 = Dense(32, name = 'weather4')(dense13)
output2 = Dense(11, name = 'output2')(dense14)

#2-3. 머지
merge1 = Concatenate()([output1, output2])    # 리스트 형태로 받아들임
merge2 = Dense(32, activation= 'swish', name = 'mg2')(merge1)
merge3 = Dense(23, activation= 'swish', name = 'mg3')(merge2)
output3= Dense(1, name = 'hidden_output')(merge3)

model = Model(inputs=[input1, input2], outputs=output3)

# model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit([samsung_x_train_split, hyundai_x_train_split], 
          samsung_y_train_split, 
          epochs = 1000, batch_size = 500,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])




#4. 평가, 예측


loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], samsung_y_test_split)
print('loss : ', loss)

result = model.predict([samsung_x_test_split, hyundai_x_test_split])

print('내일의 종가는 바로바로 : ' , result[0])


# print(samsung_x_test_split)
