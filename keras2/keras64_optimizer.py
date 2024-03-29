import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 337, train_size= 0.8, 
)

#2. 모델
model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.optimizers import Adam

learning_rate = 0.1
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss= 'mse', optimizer = optimizer)

model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print('lr : ', learning_rate, 'loss : ', results)

# lr : 0.1, loss : 1256.462646484375