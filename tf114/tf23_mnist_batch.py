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

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape)   # (60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape)     # (10000, 784) (10000, 10)

#2. 모델구성
x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float', [None, 10])

# w1 = tf.Variable(tf.random_normal([784, 128]), name= 'w1')
w1 = tf.compat.v1.get_variable('w1', shape = [784, 64])
                    # initializer = tf.contrib.layers.xavier_initializer()
b1 = tf.Variable(tf.zeros([64]), name = 'b1')
layer1 = tf.compat.v1.matmul(x, w1) + b1
dropout1 = tf.compat.v1.nn.dropout(layer1, rate = 0.3)

# w2 = tf.Variable(tf.random_normal([128, 64]), name= 'w2')
w2 = tf.compat.v1.get_variable('w2', shape = [64, 64])
                    # initializer = tf.contrib.layers.xavier_initializer()
b2 = tf.Variable(tf.zeros([64]), name = 'b2')
layer2 = tf.nn.selu(tf.compat.v1.matmul(dropout1, w2) + b2)

# w3 = tf.Variable(tf.random_normal([64, 32]), name= 'w3')
w3 = tf.compat.v1.get_variable('w3', shape = [64, 32])
                    # initializer = tf.contrib.layers.xavier_initializer()
b3 = tf.Variable(tf.zeros([32]), name = 'b3')
layer3 = tf.nn.relu(tf.compat.v1.matmul(layer2, w3) + b3)

# w4 = tf.Variable(tf.random_normal([32, 10]), name= 'w4')
w4 = tf.compat.v1.get_variable('w4', shape = [32, 10])
                    # initializer = tf.contrib.layers.xavier_initializer()
b4 = tf.Variable(tf.zeros([10]), name = 'b4')
# hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)
hypothesis = tf.matmul(layer3, w4) + b4

#3. 컴파일, 훈련
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.nn.log_softmax(hypothesis), axis = 1))  # 이거 사용할 거면 hypothesis에 softmax 빼야 성능 좋음
# loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels = y))
# loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)

train = tf.compat.v1.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)  # 위 두 줄과 같음

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600
epochs = 100

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

# tf 2.7.4 걸린시간 :  424.810986995697
# acc :  0.9204