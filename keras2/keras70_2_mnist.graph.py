# [실습] 훈련시키지 말고, 가중치든 뭐든 땡겨다가, 그래프를 그려
# epochs와 loss / acc 그래프
# 훈련하지말고 70_1에서 save한 거 땡겨와서 그려

# keras32 mnist3

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential,load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import tensorflow as tf
tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########### 실습 #############
scaler= MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

# 2. 모델구성
# model = load_model()
# model.summary()

path = './_save/pickle_test/'
import joblib
try:
    hist = joblib.load(path + 'keras70_1_history.dat')
except EOFError:
    print('EOFError 발생')


#2. 모델 - 피클 불러오기
# history 객체 로드
# import pickle
# with open('./_save/pickle_test/keras70_1_mnist_grape.pkl', 'rb') as f:
#     hist = pickle.load(f)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist['loss'], marker='.', c='red', label='loss')
plt.plot(hist['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid
plt.title('loss')
plt.xlabel('loss')
plt.ylabel('epochs')
plt.legend(loc ='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist['acc'], marker='.', c='red', label='acc')
plt.plot(hist['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid
plt.title('acc')
plt.xlabel('acc')
plt.ylabel('epochs')
plt.legend(['acc','val_acc'])

plt.show()