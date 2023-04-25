import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical   # (케라스에 원핫)

#1. 데이터
datasets = fetch_covtype()          # 오류나면 sklearn 삭제하고 다시 설치, cmd 창에 pip uninstall scikit-learn
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (581012, 54) (581012,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))        # y의 라벨값 :  [1 2 3 4 5 6 7]

######################### 원핫 #####################
y = to_categorical(y)       # y의 라벨값이 1부터 시작하는 데이터에 0이 자동으로 추가됨
y=np.delete(y, 0, axis=1)   # 앞에 자동으로 추가된 열 삭제
# print(y)
print(y.shape)          # (581012, 8)

#####################################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 2000,
    train_size= 0.8, stratify = y
)
print(y_train)
print(np.unique(y_train, return_counts= True))


#2. 모델 구성
model = Sequential()
model.add(Dense(50,activation = 'relu', input_dim = 54 ))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                   restore_best_weights = True)


model.fit(x_train, y_train, epochs = 2000, batch_size = 5000,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])


#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)

# print(y_test.shape)
# print(y_pred.shape)
# print(y_test[:5])
# print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis = 1)
y_pred = np.argmax(y_pred, axis = 1)

print(y_test_acc)
print(y_pred)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score : ', acc)


# accuracy_score :  0.8838153920294657

# accuracy_score :  0.8968443155512336