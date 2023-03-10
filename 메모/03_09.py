import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터







#2. 모델 구성
model = Sequential()
model.add(Dense(), activation = 'relu', input_dim = )
model.add(Dense(), activation = 'relu')



#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                   restore_best_weights= True)

model.fit(x_train, x_test, y_train, y_test, ephoes = 1000, batch_size= 10,
          validation_split= 0.2,
          verbose = 1,
          callbacks = [es])





#4. 평가, 예측