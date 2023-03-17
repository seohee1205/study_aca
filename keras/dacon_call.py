import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.tree import DecisionTreeClassifier

#1. 데이터
path = './_data/dacon_call/'
path_save = './_save/dacon_call/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

print(train_csv)
print(train_csv.shape)  # (30200, 13)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

print(test_csv)
print(test_csv.shape)   # (12943, 12)

# 결측치 제거
print(train_csv.info())     # 결측치 없음

x = train_csv.drop(['전화해지여부'], axis = 1)
y = train_csv['전화해지여부']

print(x.shape)      # (30200, 12)
print(y.shape)      # (30200,)

# 원핫
print('y의 라벨값 : ', np.unique(y))    # [0 1]
y = to_categorical(y)

# x와 y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 777, stratify= y
)

print(x_train.shape, x_test.shape)      # (24160, 12) (6040, 12)
print(y_train.shape, y_test.shape)      # (24160, 2) (6040, 2)

# 전처리는 데이터를 나눈 후 한다
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) 

test_csv = scaler.transform(test_csv)


#3. (함수형) 모델 구성
input1 = Input(shape=(12,))
dense1 = Dense(50,activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(37,activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(16,activation='relu')(drop2)
drop3 = Dropout(0.4)(dense3)
dense4 = Dense(10,activation='relu')(drop3)
dense5 = Dense(8,activation='relu')(dense4)
dense6 = Dense(5,activation='relu')(dense5)
output1 = Dense(2,activation='softmax')(dense6)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics= ['accuracy'])

# 정의하기
es = EarlyStopping(monitor = 'val_accuracy', patience = 1000, mode = 'auto',
                   verbose = 1,
                    restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 200,
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

f1_score = f1_score(y_test, y_predict, average ='macro')
print('f1', f1_score)

# 파일 생성
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['전화해지여부'] = y_submit
path_save = './_save/dacon_call'

submission.to_csv(path_save + 'submit_0317_0305.csv')



# result : [1.1127790212631226, 0.6117549538612366]
# acc :  0.6117549668874173