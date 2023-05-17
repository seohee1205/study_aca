# pip install keras==1.2.2
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import keras
# print(keras.__version__)
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import time

tf.compat.v1.disable_eager_execution()  # 즉시모드 안 해 1.0 
# tf.compat.v1.enable_eager_execution()  # 즉시모드 해 2.0 

# tf.set_random_seed(337) # 이거 1.x 돼
tf.random.set_seed(337) # 이거 2.x 돼

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

#[실습] 맹그러

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2], 1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape)   # (60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape)     # (10000, 784) (10000, 10)

#2. 모델구성
x = tf.compat.v1.placeholder('float', [None, 28, 28, 1])
y = tf.compat.v1.placeholder('float', [None, 10])

# 레이어1 CNN
w1 = tf.compat.v1.get_variable('w1', shape = [3, 3, 1, 64])  
       #  model.acc(Conv2D(32, kenel_size = (3, 3),input_shape=(28,28,1))) 
b1 = tf.compat.v1.Variable(tf.zeros([64]), name = 'b1')

layer1 = tf.compat.v1.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding='SAME')
layer1 += b1
L1_maxpool = tf.nn.max_pool2d(layer1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')
# (n, 14, 14, 64)

# 레이어2 CNN
w2 = tf.Variable(tf.compat.v1.random_normal([3, 3, 64, 32]), name= 'w2')
b2 = tf.Variable(tf.zeros([32]), name = 'b2')
layer2 = tf.compat.v1.nn.conv2d(L1_maxpool, w2, strides = [1, 1, 1, 1], padding='VALID')
layer2 += b2
L2_maxpool = tf.nn.max_pool2d(layer2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')
# (n, 6, 6, 32)

# 레이어3 CNN
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 3, 32, 16]), name= 'w3')
b3 = tf.Variable(tf.zeros([16]), name = 'b3')
layer3 = tf.compat.v1.nn.conv2d(L2_maxpool, w3, strides = [1, 1, 1, 1], padding='SAME')
layer3 += b3
# (n, 6, 6, 16)

# Flatten
L_flat = tf.reshape(layer3, [-1, 6*6*16])

# 레이어4 DNN
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([6*6*16, 100]), name= 'w4')
b4 = tf.Variable(tf.zeros([100]), name = 'b4')
layer4 = tf.nn.relu(tf.compat.v1.matmul(L_flat, w4) + b4)
layer4 = tf.nn.dropout(layer4, rate= 0.3)

# 레이어5 DNN
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 10]), name= 'w5')
b5 = tf.Variable(tf.zeros([10]), name = 'b5')
hypothesis = tf.nn.relu(tf.compat.v1.matmul(layer4, w5) + b5)
hypothesis = tf.nn.softmax(hypothesis)


#3. 컴파일, 훈련
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(hypothesis), axis = 1))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.nn.log_softmax(hypothesis), axis = 1))  # 이거 사용할 거면 hypothesis에 softmax 빼야 성능 좋음
# loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels = y))
# loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)

train = tf.compat.v1.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)  # 위 두 줄과 같음

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600
epochs = 20

start_time = time.time()
for step in range(epochs):
    
    avg_cost = 0
    for i in range(int(total_batch)):       # 100개씩 600번 돈다
        start = i * batch_size          # 0 , 100, 200, ...,  59900
        end = start + batch_size        # 100, 200, 300, ..., 60000

        cost_val, _, w_bal, b_val = sess.run([loss, train, w4, b4],
                      feed_dict={x:x_train[start:end], y:y_train[start:end]})    
        
        avg_cost += cost_val / total_batch
           
    print('Epoch : ', step + 1, 'loss : {:.9f}'.format(avg_cost))

end_time = time.time()
print("훈련 끝!")
        
        
#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
y_predict_arg = sess.run(tf.argmax(y_predict, 1))

y_test_arg = np.argmax(y_test, 1)

acc = accuracy_score(y_predict_arg, y_test_arg)

print("tf", tf.__version__, "걸린시간 : ", end_time - start_time)
print('acc : ', acc)
# acc :  0.8605

sess.close()

# tf 1.14.0 걸린시간 :  85.96859121322632
# acc :  0.8605