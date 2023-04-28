import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.tree import DecisionTreeClassifier

#1. 데이터
path = './_data/dacon_call/'
path_save = './_save/dacon_call/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

# print(train_csv)
# print(train_csv.shape)  # (30200, 13)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

# print(test_csv)
# print(test_csv.shape)   # (12943, 12)

# 결측치 제거
print(train_csv.isnull().sum())     # 결측치 없음

x = train_csv.drop(['전화해지여부'], axis = 1)
y = train_csv['전화해지여부']

# print(x.shape)      # (30200, 12)
# print(y.shape)      # (30200,)

# 원핫
# print('y의 라벨값 : ', np.unique(y))    # [0 1]
# y = to_categorical(y)

# x와 y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 777, stratify= y
)

# print(x_train.shape, x_test.shape)      # (24160, 12) (6040, 12)
# print(y_train.shape, y_test.shape)      # (24160, 2) (6040, 2)

# 전처리는 데이터를 나눈 후 한다
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # x_train = scaler.transform(x_train)랑 x_test = scaler.transform(x_test)랑 합친 코드
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)

test_csv = scaler.transform(test_csv)


#3. (함수형) 모델 구성
model = Sequential()
model.add(Dense(256, input_dim =12, activation= LeakyReLU(0.5)))
model.add(Dense(128, activation= LeakyReLU(0.5)))
model.add(Dropout(0.2))
model.add(Dense(128, activation= LeakyReLU(0.5)))
model.add(Dropout(0.2))
model.add(Dense(128, activation= LeakyReLU(0.5)))
model.add(Dropout(0.2))
model.add(Dense(64, activation= LeakyReLU(0.95)))
model.add(Dropout(0.2))
model.add(Dense(1, activation= 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics= ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_acc', patience = 80, mode = 'auto',
                   verbose = 1,
                    restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 2000, batch_size = 300,
          validation_split = 0.2,
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

f1_score = f1_score(y_test, y_predict, average ='macro')
print('f1', f1_score)

# 파일 생성
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['전화해지여부'] = y_submit
path_save = './_save/dacon_call/'

submission.to_csv(path_save + 'submit_0317_0450.csv')



# result : [0.32756486535072327, 0.8900662064552307]
# acc :  0.8900662251655629
# f1 0.47091800981079185