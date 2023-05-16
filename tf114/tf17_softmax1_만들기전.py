import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
tf.set_random_seed(337)

#1. 데이터
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7],
          ]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0],
          ]

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape =[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name = 'bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))     # mse

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)  # 위 두 줄과 같음

# [실습]
# 맹그러

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
       print(step, loss_val, w_val, b_val)

x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])     
y_predict = tf.compat.v1.matmul(x_test, w_val) + b_val
y_data = np.argmax(y_data, axis =1)  

y_aaa = np.argmax(sess.run(y_predict, feed_dict={x_test:x_data}), axis=1)
print(type(y_aaa))  # <class 'numpy.ndarray'>

acc = accuracy_score(y_aaa, y_data)
print('acc : ', acc)

mse = mean_squared_error(y_aaa, y_data)
print('mse : ', mse)

sess.close()

# acc :  0.125
# mse :  2.375