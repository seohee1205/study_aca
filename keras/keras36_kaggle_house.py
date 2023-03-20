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

#1-1. 결측치
print(train_csv.isnull().sum())  

df = pd.read_csv(path + 'train.csv')

# 특정 열의 결측치를 특정 값으로 대체
mean = df['LotFrontage'].mean()
df['LotFrontage'] = df['LotFrontage'].fillna(mean)

#1-2. 라벨인코딩
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)


#1-3. x, y 데이터 분리
x = train_csv.drop(['SalePrice'], axis = 1)
print(x.shape)
y = train_csv['SalePrice']
print(y.shape)

#1-4. train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.9, random_state= 888
)

print(x_train.shape, x_test.shape)      # (1008, 79) (113, 79)
print(y_train.shape, y_test.shape)      # (1008,) (113,)


#1-5. scaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)                 # fit의 범위: x_train
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
# # print(np.min(x_test), np.max(x_test))   # 0과 1이 나오는지 확인해
# test_csv = scaler.transform(test_csv)   # test_csv에도 sclaer해줘야 함


#2. 모델 
input1 = Input(shape=(79,))
dense1 = Dense(30)(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(24, activation = 'relu')(drop1)
dense3 = Dense(20, activation = 'relu')(dense2)
dense4 = Dense(12)(dense3)
dense5 = Dense(6, activation = 'relu')(dense4)
dense6 = Dense(8, activation = 'relu')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam',
              metrics= ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 80, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))
hist = model.fit(x_train, y_train, epochs = 2000, batch_size = 77,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])


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

# 시간 측정
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

y_submit = model.predict(test_csv)
y_submit = pd.DataFrame(y_submit)
y_submit = y_submit.fillna(y_submit.mean())
y_submit = np.array(y_submit)

# 파일 생성
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['SalePrice'] = y_submit
submission.to_csv(path_save + 'submit_0320_0933.csv')
