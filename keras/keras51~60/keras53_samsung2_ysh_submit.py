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
from tensorflow.keras.models import Sequential, Model, load_model
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

datasets1 = pd.read_csv(path + '삼성전자 주가3.csv', index_col=0, encoding='cp949')
datasets2 = pd.read_csv(path + '현대자동차2.csv', index_col=0, encoding='cp949')

print(datasets1, datasets2)   # [2100 rows x 16 columns]

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
    x1, x2, y, train_size=0.9, shuffle=False)

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

x1_pred = x1_test[-timesteps:].reshape(1, timesteps, 10)
x2_pred = x2_test[-timesteps:].reshape(1, timesteps, 10)


print(x1_train_split.shape)      # (165, 10, 10)
print(x2_train_split.shape)      #  (165, 10, 10)




# 모델 불러오기
model = load_model('./_save/samsung/keras53_samsung2_ysh.h5')


#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto',
#                    verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#          verbose = 1, 
#          save_best_only= True,
#          filepath="".join(['_save/samsung/keras53_samsung2_ysh.h5']))

# model.fit([x1_train_split, x2_train_split], 
#           y_train_split, 
#           epochs = 80, batch_size = 18,
#           validation_split = 0.2,
#           verbose = 1,
#           callbacks = [es])


#4. 평가, 예측

result= model.evaluate([x1_test_split,x2_test_split], y_test_split)
print('mse_loss :', result)

pred = model.predict([x1_pred, x2_pred])

print(f'마지막 날 종가 : {y[-1]} \ny_pred : {np.round(pred[0],2)}')


# model.save("./_save/samsung/keras53_samsung2_5_ysh1.h5")


# _1_ysh
# loss :  [50500100.0, 6224.962890625]
# 내일의 종가는 바로바로 :  [63758.82]

# _2_ysh
# loss :  [38813588.0, 5473.0517578125]
# 내일의 종가는 바로바로 :  [62913.72]

#_3_ysh
# mse_loss : [5113579.0, 1630.5916748046875]
# 마지막 날 종가 : 62900.0 
# y_pred : [62882.85]

#_4_ysh
# mse_loss : [4236747.5, 1835.8570556640625]
# 마지막 날 종가 : 62900.0 
# y_pred : [63459.94]

#_5_ysh
# mse_loss : [3968248.0, 1756.2916259765625]
# 마지막 날 종가 : 62900.0 
# y_pred : [63236.7]



