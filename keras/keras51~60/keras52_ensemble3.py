#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)])       # 삼성, 아모레
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])
x3_datasets = np.array([range(201, 301), range(511, 611), range(1300, 1400)])

# 온도, 습도, 강수량
print(x1_datasets.shape)    # (2, 100)
print(x2_datasets.shape)    # (3, 100)
print(x3_datasets.shape)    # (3, 100)

#1-1. 행, 열 바꾸기

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T
print(x1.shape)    # (100, 2)
print(x2.shape)    # (100, 3)
print(x3.shape)    # (100, 3)

y1 = np.array(range(2001, 2101))  # 환율
y2 = np.array(range(1001, 1101))  # 금리

# 실습1.
# concaatenate -> Concatenate로 바꿔라


#1-2. train, test 분리  ( \: 줄이 너무 길 때 씀, 한 줄이다라는 뜻)
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
    y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, x3, y1, y2, train_size = 0.7, random_state= 333 
)
# y_train, y_test = train_test_split(
#     y, train_size = 0.7, random_state= 333 
# )

print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
print(x3_train.shape, x3_test.shape)    # (70, 3) (30, 3)
print(y1_train.shape, y1_test.shape)    # (70,) (30,)
print(y2_train.shape, y2_test.shape)    # (70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape = (2,))
dense1 = Dense(35, activation = 'swish', name = 'stock1')(input1)
dense2 = Dense(24, activation = 'swish', name = 'stock2')(dense1)
dense3 = Dense(12, activation = 'swish', name = 'stock3')(dense2)
output1 = Dense(11, activation = 'swish', name = 'output1')(dense3)

#2-2. 모델2
input2 = Input(shape = (3,))
dense11 = Dense(30, name = 'weather1')(input2)
dense12 = Dense(16, activation = 'swish', name = 'weather2')(dense11)
dense13 = Dense(52, activation = 'swish', name = 'weather3')(dense12)
dense14 = Dense(32, name = 'weather4')(dense13)
output2 = Dense(11, name = 'output2')(dense14)

#2-3. 모델3
input3 = Input(shape = (3,))
dense21 = Dense(20, activation = 'swish', name = 'aaa11')(input3)
dense22 = Dense(24, activation = 'swish', name = 'aaa22')(dense21)
dense23 = Dense(10, activation = 'swish', name = 'aaa33')(dense22)
dense24 = Dense(8,  activation = 'swish', name = 'aaa44')(dense23)
output3 = Dense(11, name = 'output3')(dense24)

#2-4. 모델 합침(머지)
from tensorflow.keras.layers import concatenate, Concatenate     # 사슬처럼 잇다 / # 소문자: 함수, 대문자: class
merge1 = Concatenate()([output1, output2, output3])    # 리스트 형태로 받아들임
merge2 = Dense(32, activation= 'swish', name = 'mg2')(merge1)
merge3 = Dense(23, activation= 'swish', name = 'mg3')(merge2)
output4= Dense(18, name = 'hidden_output')(merge3)

#2-5. 분기1 
bungi1 = Dense(40,activation='swish')(output4)
bungi2 = Dense(30,activation='swish')(bungi1)
bungi3 = Dense(20,activation='swish')(bungi2)
bungi4 = Dense(10,activation='swish')(bungi3)
output5 =Dense(1)(bungi4)

#2-6. 분기2
bungi21 = Dense(30,activation='swish')(output4)
bungi22 = Dense(20,activation='swish')(bungi21)
bungi23 = Dense(10,activation='swish')(bungi22)
output6 = Dense(1)(bungi23)

model = Model(inputs = [input1, input2, input3], outputs = [output5, output6])

# 큰 모델로 봤을 때, input과 output만 맞게 하면 됨 (중간 모델의 아웃풋은 노상관~)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
import time
start = time.time()

model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs = 2000,
          batch_size = 6, validation_split = 0.2, callbacks = [es])

end = time.time()

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_squared_error

results = model.evaluate([x1_test, x2_test, x3_test],
                      [y1_test, y2_test])
print('results : ', results)

y_predict = model.predict([x1_test, x2_test, x3_test])
# np.array(y_predict = model.predict([x1_test, x2_test, x3_test]))  # np.array로 하면 shape로 볼 수 있음

print(y_predict)
# 리스트는 파이썬 기본 자료형이기 때문에 shape 함수를 사용할 수 없음, 따라서 len을 사용하여 데이터를 길이로 확인해야함
print(len(y_predict), len(y_predict[0]))    # 2, 30 / y가 몇 개인지, 0번 째 몇 개인지 

r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])
print('r2 스코어 : ', (r2_1+r2_2) / 2)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse1 = RMSE(y1_test, y_predict[0])              # RMSE 함수 사용
rmse2 = RMSE(y2_test, y_predict[1])              # RMSE 함수 사용
print("RMSE : ", (rmse1 + rmse2) / 2)

print('걸린 시간 : ', np.round(end-start, 2))


# result :  [111.13462829589844, 23.47749900817871, 87.6571273803711]
# r2 스코어 :  0.9058471294518669
# RMSE :  7.103941911444316
# 걸린 시간 :  6.58

# r2 스코어 :  0.9999970662193254
# RMSE :  0.03449732866274835
# 걸린 시간 :  17.98

# r2 스코어 :  0.9999839658432799
# RMSE :  0.09081907042644355
# 걸린 시간 :  16.29


# Concatenate 일 때
# r2 스코어 :  0.9996018571084913
# RMSE :  0.47518896881761696
# 걸린 시간 :  18.37



# 1. concatenate와 Concatenate 비교

# "concatenate"와 "Concatenate"는 모두 케라스(Keras)에서 제공하는 함수로, 
# 두 개 이상의 텐서를 결합하는 역할을 합니다.

# 그러나 "concatenate"는 이전 버전의 케라스에서 사용되던 함수로, 소문자로 작성되어 있습니다. 
# 이 함수는 두 개 이상의 텐서를 결합할 때 사용되며,
# axis 매개변수를 사용하여 결합할 축을 지정할 수 있습니다.

# 반면에 "Concatenate"는 케라스의 최신 버전에서 추가된 함수로, 대문자로 시작합니다.
# 이 함수는 "concatenate"와 동일한 역할을 수행하지만, 클래스 형태로 작성되어 있으며, 객체를 만들어 사용합니다.
# "Concatenate"는 concatenate와 달리 axis 매개변수를 사용하지 않고, axis를 지정하는 방법이 약간 다릅니다.

# 따라서, "concatenate"와 "Concatenate"는 기능적으로 동일하지만, 
# "Concatenate"는 최신 버전의 케라스에서 추가된 함수이며, 
# 보다 객체 지향적인 방식으로 사용할 수 있습니다.


# 2. 클래스 뒤에 () 왜 들어감?

# 클래스(class) 뒤에 괄호()는 해당 클래스의 인스턴스(instance)를 생성하기 위한 문법입니다. 
# 클래스는 객체(object)를 만들기 위한 청사진(blueprint)과 같은 역할을 하며, 
# 실제로 사용하기 위해서는 인스턴스를 생성해야 합니다.

# 인스턴스는 클래스를 기반으로 생성된 개별 객체를 말하며, 
# 이 객체는 클래스에서 정의한 속성(attribute)과 메서드(method)를 가지고 있습니다. 
# 이를 통해 프로그래머는 클래스에서 정의한 기능을 구현할 수 있습니다.

# 클래스 인스턴스를 생성할 때, 괄호 안에는 생성자(constructor)에 전달될 인수(arguments)를 넣을 수 있습니다.
# 생성자는 클래스에서 정의된 메서드 중 하나로, 인스턴스가 생성될 때 자동으로 호출되며, 
# 인스턴스를 초기화하는 역할을 합니다.