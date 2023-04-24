
'''
#y값 가져오기 싫을때, '빈칸 명시'하는 법 : 파이썬 기초문법 '_' 언더바 명시 
#무조건 4개를 명시해야하므로 1개면 적어서 땡겨올 수 없음 -> '_'명시해줌
#_가 변수로 먹힘(메모리 할당됨...)
(x_train, __), (x_test, _) = mnist.load_data()
# print(__.shape) #(60000,)
# print(_.shape)  #(10000,)
#이미지 데이터를 쭉 핀다면(dnn사용한다면)=? 
#=>>(70000, 784) 7만개 데이터, 784컬럼이라고 생각할 수 있음
#초반에 0이 몰려있음 => 압축시켜서 0을 줄여주는 것이 성능이 더 잘 나올 가능성 높음 
#x_train, x_test 합치기 (방법2가지)
x = np.concatenate((x_train,x_test), axis=0)  #(70000, 28, 28)
# x = np.append(x_train, x_test, axis=0)      #(70000, 28, 28)
print(x.shape) 
##########실습#######################
#pca를 통해 0.95 이상인 n_components는 몇개?
#0.95 몇개?
#0.99 몇개?
#0.999 몇개?
#1.0 몇개?
#Hint : np.argmax
#####################################
#reshape (pca는 2차원만 받으므로)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])  #(70000,28*28)
print(x.shape) #(70000, 784)
#데이터x 컬럼 축소
pca = PCA(n_components=784)  #[154,331,486,713]
x = pca.fit_transform(x)
#설명가능한 변화율
# pca_EVR = pca.explained_variance_ratio_  
# cumsum = np.cumsum(pca_EVR)  #배열의 누적합
# print(cumsum)
# print(np.argmax(cumsum >= 0.95) +1) #154
# print(np.argmax(cumsum >= 0.99) +1) #331
# print(np.argmax(cumsum >= 0.999) +1) #486
# print(np.argmax(cumsum >= 1.0) +1) #712나옴 -> 0부터 시작하므로 +1해줘야 713개 나옴
'''

#[실습]##################################################################################
#모델 만들어서 비교하기(dnn최상모델 가져오기) 
#                    *acc값*
#1. 나의 최고의 CNN:  가장 좋음
#2. 나의 최고의 DNN:  dnn과 비교(pca 몇 일때 가장 좋은지 )
#3. PCA 0.95      :
#3. PCA 0.99      :
#3. PCA 0.999     :
#3. PCA 1.0       :
########################################################################################

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import pandas as pd
from sklearn.metrics import accuracy_score

# 1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))

cum_list = ['154', '331', '486', '713']
cum_name_list = ['pca 0.95', 'pca 0.99', 'pca 0.999', 'pca 1.0']

result_list = []
acc_list = []
for i in range(len(cum_list)):
    pca = PCA(n_components=int(cum_list[i]))
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # 모델 구성
    model = Sequential()
    model.add(Dense(64, input_shape=(int(cum_list[i]),)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # 모델 훈련
    hist = model.fit(x_train_pca, y_train, 
                     epochs=50, 
                     batch_size=5000, 
                     verbose=1,
                     validation_split=0.2)

    # 모델 평가
    result = model.evaluate(x_test_pca, y_test)
    y_predict = model.predict(x_test_pca)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
    
    result_list.append(result)
    acc_list.append(acc)

for i in range(len(cum_list)):
    print(f"{cum_name_list[i]}: loss={result_list[i]:.4f}, acc={acc_list[i]:.4f}")

print("나의 최고의 CNN : 0.9816")
print("나의 최고의 DNN : 0.954")
    


# pca 0.95: loss=0.7444, acc=0.8519
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# pca 0.99: loss=0.7678, acc=0.8533
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# pca 0.999: loss=0.9679, acc=0.8013
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# pca 1.0: loss=0.7280, acc=0.8822
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954

# acc :  0.9141
# accuracy_score :  0.9141