import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 같으면 0 다르면 1

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim= 2, activation= 'sigmoid'))


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer= 'adam',
              metrics= ['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs= 100)

#4. 평가, 예측
y_predict = model.predict(x_data)


# results = model.score(x_data, y_data)
results = (model.evaluate(x_data, y_data))
print("model.score : ", results[1])

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_score : ", acc)
