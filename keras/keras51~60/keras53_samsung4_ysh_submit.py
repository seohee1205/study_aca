# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞히기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용금지

#마감시간 : 27일 월 23시 59분 59초        /    28일 화 23시 59분 59초
#윤서희 [현대 2차] ?원
#첨부파일 : keras53_samsung2_ysh_submit.py       데이터 및 가중치 불러오는 로드가 있어야함
#          keras53_samsung4_ysh_submit.py
#가중치 :  _save/samsung/keras53_samsung2_ysh.h5 / hdf5
#         _save/samsung/keras53_samsung4_ysh.h5 / hdf5
# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞히기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용금지

#마감시간 : 27일 월 23시 59분 59초        /    28일 화 23시 59분 59초
#윤서희 [현대 2차] ?원
#첨부파일 : keras53_samsung2_ysh_submit.py       데이터 및 가중치 불러오는 로드가 있어야함
#          keras53_samsung4_ysh_submit.py
#가중치 :  _save/samsung/keras53_samsung2_ysh.h5 / hdf5
#         _save/samsung/keras53_samsung4_ysh.h5 / hdf5


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, concatenate, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.linear_model import LinearRegression

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

y = datasets2['시가']

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)

x1 = x1[:200]
x2 = x2[:200]
y = y[:200]

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


timesteps=15

def split_x(datasets, timesteps):
    aaa=[]
    for i in range(len(datasets)-timesteps-1):
        subset = datasets[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x1_train_split = split_x(x1_train, timesteps)
x1_test_split = split_x(x1_test, timesteps)
x2_train_split = split_x(x2_train, timesteps)
x2_test_split = split_x(x2_test, timesteps)

y_train_split = y_train[(timesteps+1) : ]
y_test_split = y_test[(timesteps+1) : ]


x1_pred = x1_test[-timesteps:].reshape(1, timesteps, 10)
x2_pred = x2_test[-timesteps:].reshape(1, timesteps, 10)


print(x1_train_split.shape)    # (165, 15, 10)
print(x2_train_split.shape)    # (165, 15, 10)


# 모델 불러오기
model = load_model('./_save/samsung/keras53_samsung4_ysh.h5')


#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# es = EarlyStopping(monitor = 'val_loss', patience = 200, mode = 'auto',
#                    verbose = 1, restore_best_weights= True)

# # mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
# #          verbose = 1, 
# #          save_best_only= True,
# #          filepath="".join(['_save/samsung/keras53_samsung2_ysh.h5']))

# model.fit([x1_train_split, x2_train_split], 
#           y_train_split, 
#           epochs = 2000, batch_size = 22,
#           validation_split = 0.2,
#           verbose = 1,
#           callbacks = [es])


#4. 평가, 예측

result= model.evaluate([x1_test_split,x2_test_split], y_test_split)
print('loss :', result)

predict_result = model.predict([x1_pred, x2_pred])

print(f'어제 시가는 : {y[-1]} \n이틀 뒤의 시가는 바로 : {np.round(predict_result[0],2)}')


# model.save("_save/samsung/keras53_samsung4_12_ysh.h5")


# 1
# loss : [146148368.0, 10497.6171875]
# 어제 시가는 : 177100.0
# 이틀 뒤의 시가는 바로 : [177442.8]

#2 
# loss : [225892992.0, 13420.75]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [177352.12]

#3
# loss : [391196960.0, 18545.671875]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [174059.4]

#4
# loss : [264029056.0, 14686.81640625]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [173540.12]

#5
# loss : [124958512.0, 9755.2734375]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [174194.52]

#6
# loss : [274485088.0, 16182.97265625]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [175027.3]

#7
# loss : [128302784.0, 11073.94140625]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [175387.69]

#8
# loss : [55588232.0, 6270.5078125]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [180310.06]

#9
# loss : [271741056.0, 13240.01953125]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [179224.16]

#10
# loss : [112393976.0, 8893.16015625]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [181306.9]

#11
# loss : [57661980.0, 6835.16015625]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [181082.55]

#12
# loss : [117850232.0, 9447.87890625]
# 어제 시가는 : 177100.0 
# 이틀 뒤의 시가는 바로 : [179253.]

