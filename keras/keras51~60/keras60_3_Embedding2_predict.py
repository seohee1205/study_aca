from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape

#1. 데이터
docs = ['너무 재밌어요', '참 최고예요', ' 참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요']

x_predict = ['나는 성호가 정말 재미없다 너무 정말']

# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0])

data = docs + x_predict
print(data)

token = Tokenizer()
token.fit_on_texts(data)

# print(token.word_index)
# print(token.word_counts)

x = token.texts_to_sequences(data)      # 수치화
print(x)
# [[1, 7], [2, 8], [2, 3, 9, 10], [11, 12, 13], [14, 15, 16, 17, 18], 
# [19], [20], [21, 22], [23, 24], [25], [1, 4], [2, 26], [5, 3, 27, 28], 
# [5, 29], [30, 31, 6, 4, 1, 6]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding= 'pre', maxlen= 5)
print(pad_x.shape)  # (15, 5)

pad_x_train = pad_x[:14, :]
pad_x_pred = pad_x[14, :]
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_pred = pad_x_pred.reshape(1, 5, 1)


word_index = len(token.word_index)
print("단어사전의 갯수 : ", word_index)     # 단어사전의 갯수 :  31


#2. 모델 
model = Sequential()
# model.add(Dense(64, input_shape = (5,)))
model.add(Reshape(target_shape = (5, 1), input_shape=(5,)))
model.add(LSTM(32))
model.add(Dense(32, activation = 'relu' ))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

model.fit(pad_x_train, labels, epochs = 100, batch_size = 18)


#4. 평가, 예측
acc = model.evaluate(pad_x_train, labels)[1]
print('acc : ', acc)

y_pred = model.predict(pad_x_pred)
print(y_pred)



########################## [실습] ############################
# 긍정인지 부정인지 맞혀봐

print(pad_x[0])
x_pred= pad_x[0].reshape(1, 5, 1)
y_pred_1 = np.round(model.predict(x_pred))
print(y_pred_1)

# acc :  0.8571428656578064
# [[0.99966323]]
# [0 0 0 1 7]
# [[1.]]