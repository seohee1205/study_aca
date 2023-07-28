# 맹그러
# 가중치 동결과 동결하지 않았을 때, 그리고 원래와 성능 비교
# Flatten과 GAP 차이

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
tf.random.set_seed(337)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = VGG16()   # include_top True, input_shape = (224, 224, 3)     # 디폴트 상태
vgg16 = VGG16(weights= 'imagenet', include_top=False,
              input_shape=(32, 32, 3)
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(10, activation = 'softmax'))

# model.trainable = True

model.summary()

print(len(model.weights))               
print(len(model.trainable_weights))     

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

### Flatten
# False (동결)
# loss : 1.2040126323699951
# acc 0.5855000019073486
# acc : 0.5855
# 걸린 시간: 1분 44초

# True (동결 X)
# loss : 0.7726275324821472
# acc 0.7818999886512756
# acc : 0.7819
# 걸린 시간: 6분 31초

## GlobalAveragePooling2D
# False (동결)
# loss : 1.2179967164993286
# acc 0.5813999772071838
# acc : 0.5814
# 걸린 시간: 22분 54초


# True (동결 X)