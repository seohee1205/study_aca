from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score 
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


print(np.unique(y_train,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.
x_test = x_test / 255.


base_model = VGG16(weights='imagenet', include_top= False,
                   input_shape=(32, 32, 3))
# print(base_model.output)
# KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), 
#                                  dtype=tf.float32, name=None), 
#             name='block5_pool/MaxPool:0', description="created by layer 'block5_pool'")

x = base_model.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(10, activation= 'softmax')(x)

model = Model(inputs = base_model.input, outputs = output1)

model.summary()

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
