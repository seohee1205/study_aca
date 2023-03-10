from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 123, test_size= 0.2
)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 5,
                 validation_split = 0.2, verbose = 1)
print("=============================================")
print(hist)
# <tensorflow.python.keras.callbacks.History object at 0x00000227C99A01F0>
print("=============================================")
print(hist.history)
print("=============================================")
print(hist.history['loss'])
print("=============================================")
print(hist.history['val_loss'])
print("=============================================")


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()



# val_loss가 loss보다 높은 위치에 있음