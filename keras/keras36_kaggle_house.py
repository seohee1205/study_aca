from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, LabelEncoder
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv)
print(train_csv.shape)  # (1460, 80)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

print(test_csv)
print(test_csv.shape)   # (1459, 79)

# 결측치 제거
print(train_csv.isnull().sum())     # LotFrontage

x = train_csv.drop(['SalePrice'], axis = 1)
print(x)
y = train_csv['SalePrice']
print(y)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 789
)

print(x_train.shape, x_test.shape)      # (1168, 79) (292, 79)
print(y_train.shape, y_test.shape)      # (1168,) (292,)


##################################
from sklearn.preprocessing import OrdinalEncoder

print(np.unique(train_csv['MasVnrType']))


'''
data = pd.DataFrame({
    'MSZoning': ['RH', 'RL', 'RM', 'FV', 'C (all)']
    'Street': ['Pave', 'Grvl', 'apple'],
    'Alley': ['NA', 'Pave', 'Grvl']
    'LotShape': ['Reg', 'IR1', 'RM', 'FV', 'C (all)']
    'LandContour': ['Lvl', 'HLS', 'Bnk', 'Low', 'C (all)']
    'Utilities': ['AllPub', 'NoSeWa']
    'LotConfig': ['Corner' 'CulDSac' 'FR2' 'FR3' 'Inside']
    'LandSlope': ['Gtl' 'Mod' 'Sev']
    'Neighborhood': ['Blmngtn' 'Blueste' 'BrDale' 'BrkSide' 'ClearCr' 'CollgCr' 'Crawfor'
    'Edwards' 'Gilbert' 'IDOTRR' 'MeadowV' 'Mitchel' 'NAmes' 'NPkVill'
    'NWAmes' 'NoRidge' 'NridgHt' 'OldTown' 'SWISU' 'Sawyer' 'SawyerW'
    'Somerst' 'StoneBr' 'Timber' 'Veenker']
    'Condition1': ['Artery' 'Feedr' 'Norm' 'PosA' 'PosN' 'RRAe' 'RRAn' 'RRNe' 'RRNn']
    # 10개
    
    'Condition2': ['Artery' 'Feedr' 'Norm' 'PosA' 'PosN' 'RRAe' 'RRAn' 'RRNn']
    'BldgType': ['1Fam' '2fmCon' 'Duplex' 'Twnhs' 'TwnhsE']
    'HouseStyle': ['1.5Fin' '1.5Unf' '1Story' '2.5Fin' '2.5Unf' '2Story' 'SFoyer' 'SLvl']
    'RoofStyle': 'Flat' 'Gable' 'Gambrel' 'Hip' 'Mansard' 'Shed']
    'RoofMatl': ['ClyTile' 'CompShg' 'Membran' 'Metal' 'Roll' 'Tar&Grv' 'WdShake' 'WdShngl']
    'Exterior1st': ['AsbShng' 'AsphShn' 'BrkComm' 'BrkFace' 'CBlock' 'CemntBd' 'HdBoard'
 'ImStucc' 'MetalSd' 'Plywood' 'Stone' 'Stucco' 'VinylSd' 'Wd Sdng'
 'WdShing']
    'Exterior2nd': ['AsbShng' 'AsphShn' 'Brk Cmn' 'BrkFace' 'CBlock' 'CmentBd' 'HdBoard'
 'ImStucc' 'MetalSd' 'Other' 'Plywood' 'Stone' 'Stucco' 'VinylSd'
 'Wd Sdng' 'Wd Shng']
   col26 'MasVnrType': []
    'RoofMatl': []
    'RoofMatl': []
    ####### 10개
    
    'RoofMatl': []
    'RoofMatl': []
    
    
    
    
})
'''





# # 분리형을 수치형으로 바꿔주기
# le = LabelEncoder()
# le.fit(train_csv['MSSubClass'])   # 0과 1로 인정
# aaa = le.transform(train_csv['MSSubClass'])
# print(aaa)
# print(type(aaa))    # <class 'numpy.ndarray'>
# print(aaa.shape)    # (5497,) 벡터 형태
# print(np.unique(aaa, return_counts= True))      # 몇 개씩 있는지    (array([0, 1]), array([1338, 4159], dtype=int64)) => 0이 1338개, 1이 4159개

# train_csv['MSSubClass'] = aaa
# print(train_csv)
# test_csv['MSSubClass'] = le.transform(test_csv['MSSubClass'])




'''
scaler = RobustScaler()
scaler.fit(x_train)                 # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
# print(np.min(x_test), np.max(x_test))   # 0과 1이 나오는지 확인해

test_csv = scaler.transform(test_csv)   # test_csv에도 sclaer해줘야 함



# #2. 모델 
model = Sequential()
model.add(Dense(20, input_dim = 8))
model.add(Dropout(0.3))
model.add(Dense(18, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam',
              metrics= ['accuracy'])

import datetime
date = datetime.datetime.now()
print(date)  # 2023-03-14 11:11:30.046663
date = date.strftime("%m%d_%H%M") # 시간을 문자로 (월, 일, 시간, 분)
print(date)  # 0314_1116

filepath = './_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # epoch의 4번째 정수까지, val-loss의 4번째 소수까지

# 정의하기
es = EarlyStopping(monitor = 'val_accuracy', patience = 35, mode = 'auto',
                   verbose = 1,
                    restore_best_weights = True)

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
        verbose = 1, 
        save_best_only= True,
        filepath="".join([filepath, 'k27_', date, '_', filename]))

hist = model.fit(x_train, y_train, epochs = 5000, batch_size = 30,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es, mcp])


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

# 파일 생성
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

path_save = './_save/kaggle_house/'
submission.to_csv(path_save + 'submit_0317_0845.csv')

'''