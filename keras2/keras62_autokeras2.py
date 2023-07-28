import autokeras as ak
import tensorflow as tf
from keras.datasets import mnist
import time
# print(ak.__version__) #1.0.20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# autokeras 모델 생성
model = ak.ImageClassifier(max_trials=1,
                         overwrite=False)  # 최대 시도 횟수 지정

path = './_save/autokeras/'
# best_model.save(path + "keras62_autokeras1.h5") 
model = tf.keras.models.load_model([path + "keras62_autokeras1.h5"])


# 최적의 모델 출력
print(model.summary())   

#4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과 : ', results)


