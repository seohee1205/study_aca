import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (581012, 54) (581012,)

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

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor= 'val_loss', patience=20, mode= 'min', verbose= 1)
rlr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto', verbose = 1, 
                        factor= 0.5)
model.fit(x_train, y_train, epochs = 200, batch_size = 32, verbose=1, validation_split= 0.2,
          callbacks=[es, rlr])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print('lr : ', learning_rate, 'loss : ', results)

# lr :  0.1 loss :  7400531.0
