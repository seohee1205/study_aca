# 분류: 이진과 다중으로 나뉨
# 과제: 리스트, 딕셔너리, 튜플 세 가지에 대해서 공부하고 어떤 건지 메일로 제출

import numpy as np
from sklearn.datasets import load_breast_cancer     # 유방암 데이터 (암에 걸렸냐 안 걸렸냐)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)           # *** 판다스 : .describe()
print(datasets.feature_names)   # *** 판다스 : .columns()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)     # (569, 30) (569,)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 555, test_size= 0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, activation = 'relu', input_dim = 30))
model.add(Dense(5, activation = 'linear'))
model.add(Dense(8, activation = 'linear'))
model.add(Dense(5, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))  # 마지막 레이어 1개인 이유: (569, ) 때문에 569 스칼라가 모인 벡터 1개이기 때문
# 시그모이드: 0에서 1로 한정되게 하기 위해 사용
# 이진수 -> 모델 마지막 액티베이션을 시그모이드로,
# 컴파일 loss 부분을 binary_crossentropy로 함 (mse로 쓰면 실수로 나오기 때문)

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics= ['accuracy', 'acc', 'mse'], #'mean_suquared_error'] # 결과에 mse, mae도 보고 싶으면 metrics에 추가하면 됨.
              ) # 'accuracy' = 'acc'

# 정의하기 (얼리스탑핑)
es = EarlyStopping(monitor = 'val_loss', patience = 25, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)

hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 25,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('암')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)   # results :  [0.13558533787727356, 0.9649122953414917, 0.034711625427007675] => loss, accuracy, mse 순

y_predict = np.round(model.predict(x_test)) #np.round: 0과 1로 나온 걸 이용하게끔
# print("=======================================")
# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("=======================================")
# 50~54 결과: 
# [1 0 1 1 0]
# [[ 0.7635741 ]
#  [-0.06654218]
#  [ 0.84689856]
#  [ 0.62947226]
#  [-0.96417093]]
# 설명: mse를 써주면 무조건 실수로 빼기 때문에 사용할 수 없다(0과 0.000001은 다름, 반올림 후 수정)
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)    # acc: 0.956140350877193



