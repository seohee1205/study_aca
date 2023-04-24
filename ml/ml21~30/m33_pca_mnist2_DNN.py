from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D,Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score 


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
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.decomposition import PCA

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 입력 데이터를 2차원에서 1차원으로 변경
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 클래스 레이블을 one-hot 인코딩
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 데이터 축소
n_components = [154, 331, 486, 784]
x_train_pca = []
x_test_pca = []

for i in n_components:
    pca = PCA(n_components=i)
    pca.fit(x_train)
    x_train_pca.append(pca.transform(x_train))
    x_test_pca.append(pca.transform(x_test))

# DNN 모델 학습 및 평가
for i in range(len(n_components)):
    model = Sequential()
    model.add(Dense(512, input_dim=n_components[i], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pca[i], y_train, epochs=10, batch_size=256, validation_data=(x_test_pca[i], y_test))
    _, accuracy = model.evaluate(x_test_pca[i], y_test, verbose=0)
    
    print("나의 최고의 CNN : 0.9816")
    print("나의 최고의 DNN : 0.954")
    print(f"PCA {n_components[i]/784.0:.3f} test accuracy: {accuracy:.4f}")

    
# Epoch 1/10
# 2023-04-24 19:29:40.977457: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
# 235/235 [==============================] - 3s 6ms/step - loss: 2.2545 - accuracy: 0.8640 - val_loss: 0.4300 - val_accuracy: 0.8979
# Epoch 2/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.2411 - accuracy: 0.9402 - val_loss: 0.2922 - val_accuracy: 0.9337
# Epoch 3/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.1268 - accuracy: 0.9657 - val_loss: 0.2775 - val_accuracy: 0.9450
# Epoch 4/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.0767 - accuracy: 0.9768 - val_loss: 0.3141 - val_accuracy: 0.9482
# Epoch 5/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0626 - accuracy: 0.9809 - val_loss: 0.2489 - val_accuracy: 0.9566
# Epoch 6/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0464 - accuracy: 0.9852 - val_loss: 0.2470 - val_accuracy: 0.9585
# Epoch 7/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0482 - accuracy: 0.9854 - val_loss: 0.2574 - val_accuracy: 0.9575
# Epoch 8/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0448 - accuracy: 0.9874 - val_loss: 0.2677 - val_accuracy: 0.9596
# Epoch 9/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0586 - accuracy: 0.9838 - val_loss: 0.2841 - val_accuracy: 0.9553
# Epoch 10/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0661 - accuracy: 0.9835 - val_loss: 0.2612 - val_accuracy: 0.9616
############################################################
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# PCA 0.196 test accuracy: 0.9616
#############################################################
# Epoch 1/10
# 235/235 [==============================] - 2s 7ms/step - loss: 2.5124 - accuracy: 0.8769 - val_loss: 0.5618 - val_accuracy: 0.9275
# Epoch 2/10
# 235/235 [==============================] - 2s 6ms/step - loss: 0.2445 - accuracy: 0.9571 - val_loss: 0.3934 - val_accuracy: 0.9432
# Epoch 3/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0974 - accuracy: 0.9777 - val_loss: 0.3928 - val_accuracy: 0.9460
# Epoch 4/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0475 - accuracy: 0.9875 - val_loss: 0.4035 - val_accuracy: 0.9477
# Epoch 5/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0361 - accuracy: 0.9900 - val_loss: 0.3656 - val_accuracy: 0.9556
# Epoch 6/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0474 - accuracy: 0.9881 - val_loss: 0.4071 - val_accuracy: 0.9541
# Epoch 7/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0599 - accuracy: 0.9865 - val_loss: 0.4024 - val_accuracy: 0.9527
# Epoch 8/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0863 - accuracy: 0.9837 - val_loss: 0.3959 - val_accuracy: 0.9552
# Epoch 9/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0963 - accuracy: 0.9819 - val_loss: 0.4007 - val_accuracy: 0.9567
# Epoch 10/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0574 - accuracy: 0.9877 - val_loss: 0.3771 - val_accuracy: 0.9593
#############################################################
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# PCA 0.422 test accuracy: 0.9593
#############################################################
# Epoch 1/10
# 235/235 [==============================] - 2s 7ms/step - loss: 2.0712 - accuracy: 0.8902 - val_loss: 0.5481 - val_accuracy: 0.9343
# Epoch 2/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.2380 - accuracy: 0.9614 - val_loss: 0.4584 - val_accuracy: 0.9442
# Epoch 3/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.1017 - accuracy: 0.9803 - val_loss: 0.4028 - val_accuracy: 0.9529
# Epoch 4/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0622 - accuracy: 0.9863 - val_loss: 0.4034 - val_accuracy: 0.9553
# Epoch 5/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.0536 - accuracy: 0.9883 - val_loss: 0.4137 - val_accuracy: 0.9565
# Epoch 6/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0567 - accuracy: 0.9883 - val_loss: 0.4721 - val_accuracy: 0.9556
# Epoch 7/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0873 - accuracy: 0.9845 - val_loss: 0.4961 - val_accuracy: 0.9564
# Epoch 8/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0926 - accuracy: 0.9842 - val_loss: 0.4875 - val_accuracy: 0.9560
# Epoch 9/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0872 - accuracy: 0.9843 - val_loss: 0.3875 - val_accuracy: 0.9621
# Epoch 10/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0568 - accuracy: 0.9891 - val_loss: 0.3697 - val_accuracy: 0.9648
#############################################################
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# PCA 0.620 test accuracy: 0.9648
#############################################################
# Epoch 1/10
# 235/235 [==============================] - 2s 7ms/step - loss: 1.6039 - accuracy: 0.8855 - val_loss: 0.4081 - val_accuracy: 0.9283
# Epoch 2/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.1705 - accuracy: 0.9629 - val_loss: 0.3187 - val_accuracy: 0.9443
# Epoch 3/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.0667 - accuracy: 0.9819 - val_loss: 0.3029 - val_accuracy: 0.9519
# Epoch 4/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0369 - accuracy: 0.9891 - val_loss: 0.2859 - val_accuracy: 0.9562
# Epoch 5/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0308 - accuracy: 0.9909 - val_loss: 0.2957 - val_accuracy: 0.9581
# Epoch 6/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0493 - accuracy: 0.9874 - val_loss: 0.3515 - val_accuracy: 0.9542
# Epoch 7/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.0694 - accuracy: 0.9834 - val_loss: 0.3257 - val_accuracy: 0.9577
# Epoch 8/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0703 - accuracy: 0.9847 - val_loss: 0.3740 - val_accuracy: 0.9521
# Epoch 9/10
# 235/235 [==============================] - 1s 6ms/step - loss: 0.0598 - accuracy: 0.9866 - val_loss: 0.3295 - val_accuracy: 0.9618
# Epoch 10/10
# 235/235 [==============================] - 1s 5ms/step - loss: 0.0494 - accuracy: 0.9884 - val_loss: 0.3183 - val_accuracy: 0.9623
#############################################################
# 나의 최고의 CNN : 0.9816
# 나의 최고의 DNN : 0.954
# PCA 1.000 test accuracy: 0.9624
#############################################################
              