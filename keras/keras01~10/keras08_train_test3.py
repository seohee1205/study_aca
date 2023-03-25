import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법
# 힌트 : 사이킷런

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size = 0.7,
    test_size = 0.3,
    random_state= 1234,  # 데이터가 바뀌면 모델의 성능을 알 수 없기 때문에, 데이터 값을 랜덤으로 뽑아냈더라도 두 번째도 데이터를 똑같이 하기 위해서 하는 것이 randm_state
    shuffle = True      # 랜덤으로 데이터 뽑기
    )                        

print(x_train)
print(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([11])
print('[11]의 예측값 : ', result)

# (1) 
# 10, 5, 7, 3, 1 / epochs = 100, 7/7 [==============================] - 0s 973us/step - loss: 0.0016
# 1/1 [==============================] - 0s 103ms/step - loss: 0.0018
# loss :  0.001771476469002664
# 1/1 [==============================] - 0s 68ms/step

