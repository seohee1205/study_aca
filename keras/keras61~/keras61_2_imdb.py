from keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Embedding, Flatten, Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words= 10000
)

# print(x_train)
# print(y_train)      # [1 0 0 ... 0 1 0]
print(x_train.shape, x_test.shape)   # (25000,) (25000,)
print(np.unique(y_train, return_counts= True))    # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
# np.unique = 판다스에선 value_counts
print(pd.value_counts(y_train)) 
# 1    12500
# 0    12500

print("영화평의 최대길이 : ", max(len(i) for i in x_train))   # 2494
print("영화평의 평균길이 : ", sum(map(len, x_train)) / len(x_train))  #  238.71364

# [실습]

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x_train = pad_sequences(x_train, padding= 'pre', maxlen = 100, truncating= 'pre')   #  앞을 잘라 버리겠다
pad_x_test = pad_sequences(x_test, padding= 'pre', maxlen = 100, truncating= 'pre')

print(x_train.shape, x_test.shape)    # (25000,) (25000,)

# softmax 46, embedding input_dim = 10000, output_dim = 마음대로, input_length = max(len)
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_test = pad_x_test.reshape(pad_x_test.shape[0], pad_x_test.shape[1], 1)


# 2. 모델
model = Sequential()
model.add(Embedding(10000, 32, input_shape=(100,)))     # input_length 명시해야 Flatten 사용가능 
model.add(Conv1D(84, 2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor = 'acc', patience = 80, mode = 'auto',
                   verbose = 1,
                    restore_best_weights = True)
model.fit(pad_x_train, y_train, epochs=1000, batch_size=40, callbacks = [es])

# 4. 평가, 예측
acc = model.evaluate(pad_x_test, y_test)[1]
print('acc : ', acc)


# acc :  0.8122400045394897