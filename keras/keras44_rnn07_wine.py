from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Input, Dropout, MaxPooling2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pandas as pd
import time

#1. 데이터
datasets=load_wine()

# print(train_csv.shape) #(10886, 11)
x = datasets.data
y = datasets['target']

# print(np.unique(y, return_counts=True))
# print(x.shape, y.shape) #(150, 4) (150,)

y = pd.get_dummies(y)
print(y.shape)

y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=221,
                                                    stratify=y
                                                    )

print(x_train.shape, x_test.shape)  #(142, 13) (36, 13)


scaler=MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))
# print(x_train.shape, x_test.shape)  
x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)


#2. 모델
input1 = Input(shape=(13, 1))
LSTM1 = LSTM(64, activation='linear')(input1)
dense1 = Dense(26, activation='relu')(LSTM1)
dense2 = Dense(16, activation='relu')(dense1)
dense3 = Dense(12, activation='relu')(dense2)
output1 = Dense(3, activation = 'softmax')(dense3)
model = Model(inputs=input1, outputs=output1)


# model.summary()

#3. 컴파일, 훈련
start =time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam')
es = EarlyStopping(monitor = 'val_loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1)
hist = model.fit(x_train, y_train,
                 epochs =1000,
                 batch_size=16,
                 validation_split = 0.2,
                 callbacks=[es])

end =time.time()


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)
print('걸린 시간 : ', np.round(end-start, 2))


# result : 0.2626489996910095
# acc : 0.9444444444444444
# 걸린 시간 :  15.78
