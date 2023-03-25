from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 650, train_size= 0.7
)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 25, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    # val-loss를 기준으로 할 것이고, 5번 참을 것이다. / val-loss의 최솟값을 찾아라
              restore_best_weights= True)  # 최적(최소 loss)의 w 값을 반환한다.


hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 30,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )


# print("=============================================")
# print(hist)
# # <tensorflow.python.keras.callbacks.History object at 0x00000227C99A01F0>
# print("=============================================")
# print(hist.history)
# print("=============================================")
# print(hist.history['loss'])
# print("======================발로스=======================")
# print(hist.history['val_loss'])
# print("======================발로스=======================")


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


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 스코어 : ', r2)




# r2 스코어 :  0.6599571014467418
# r2 스코어 :  0.569996514407694
# r2 스코어 :  0.7244894812227904

#loss :  33.218955993652344
# r2 스코어 :  0.3549752183660997

### loss :  13.246586799621582
### r2 스코어 :  0.8065346116576226


