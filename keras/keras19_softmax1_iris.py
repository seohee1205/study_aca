import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.utils import to_categorical   # (케라스에 원핫)

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)   # 판다스에서 describe() 와 동일
print(datasets.feature_names) # 판다스에서 clolumns 와 동일
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (150, 4) (150,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))    # y의 라벨값 :  [0 1 2]

################# 이 지점에서 원핫을 해줘야 함 #######################
from tensorflow.keras.utils import to_categorical   # (케라스에 원핫)
y = to_categorical(y)
# print(y)
print(y.shape)      # (150, 3)

## y를 (150, ) -> (150, 3)

# 판다스에 겟더미




# 사이킷런에 원핫인코더

#####################################################################


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 333, 
    train_size = 0.8, 
    stratify = y        # 데이터를 일정 비율로 분배해라 
)
print(y_train)
print(np.unique(y_train, return_counts= True))

#2. 모델 구성
model = Sequential()
model.add(Dense(50, activation = 'relu', input_dim = 4))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax')) # 3개의 확률값의 합 = 1
# 다중분류 문제는 마지막 레이어의 activation = softmax, 주의: 모델 마지막 부분(output)은 y의 라벨값의 개수만큼

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics= ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)

model.fit(x_train, y_train, epochs = 50, batch_size = 10,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

'''
#4. 평가, 예측
y_predict=model.predict(x_test)
y_predict=np.array(y_predict)
print(y_predict)
print("-----------")
print(y_test)

y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
'''


# [실습] accuracy_score를 사용해서 스코어를 빼세요.
# acc :  0.9666666666666667
###############################################################################
#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)
      
# print(y_test.shape)         # (30, 3)
# print(y_pred.shape)         # (30, 3)
# print(y_test[:5])       
# print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis = 1)    # 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis = 1)

print(y_test_acc)
print(y_pred)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score : ', acc)

