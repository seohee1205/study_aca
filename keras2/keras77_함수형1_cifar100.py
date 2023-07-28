# 함수형 맹그러봐

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np


#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)


print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.
x_test = x_test / 255.

input1 = Input(shape = (32, 32, 3))
vgg16 = VGG16(include_top=False, weights = 'imagenet')(input1)
# gap1 = GlobalAveragePooling2D()(vgg16)
flt1 = Flatten()(vgg16)
# output1 = Dense(10, activation = 'softmax')(gap1)
output1 = Dense(100, activation = 'softmax')(flt1)

model = Model(inputs=input1, outputs=output1)

model.summary()

##################### 글로벌 에버뤼지 연산량 ######################
# vgg16 (Functional)          (None, None, None, 512)   14714688

# global_average_pooling2d (G  (None, 512)              0
# lobalAveragePooling2D)

# dense (Dense)               (None, 10)                5130

# =================================================================
# Total params: 14,719,818
# Trainable params: 14,719,818
# Non-trainable params: 0

# 3. 컴파일, 훈련
hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='max', patience=100, verbose=1, restore_best_weights=True)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()

# 걸린 시간 계산
elapsed_time = end - start

# 분과 초로 변환
minutes = elapsed_time // 60
seconds = elapsed_time % 60

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)
print('loss :', result[0])
print('acc', result[1])
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
print(f'acc : {acc}')

# 출력
print("걸린 시간: {}분 {}초".format(int(minutes), int(seconds)))
