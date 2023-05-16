import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(337)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.loat32)

# [실습] 맹그러봐
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

#2. 모델
# model.add(Dense(10, input_shape=2))
w1 = tf.Variable(tf.compat.v1.random_normal([2, 10]), name = 'weight1')
b1 = tf.Variable(tf.compat.v1.zeros([10]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1)+ b1

# model.add(Dense(7))
w2 = tf.Variable(tf.compat.v1.random_normal([10, 7]), name = 'weight2')
b2 = tf.Variable(tf.compat.v1.zeros([7]), name='bias2')
layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer1, w2)+ b2)

# model.add(Dense(1, activation= 'sigmoid'))
w3 = tf.Variable(tf.compat.v1.random_normal([7, 1]), name = 'weight3')
b3 = tf.Variable(tf.compat.v1.zeros([1]), name='bias3')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3)+ b3)


#3-1. 컴파일
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)    # cast -> True 아니면 False
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
# accuracy = false, True로 반환 -> float32 1.0, 2.0 이렇게 됨 -> 숫자의 나누기 4해서 0.5가 됨


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1501):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data , y:y_data})

        if step % 200 == 0:
            print(step, cost_val)
            
    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x:x_data, y:y_data})
    print("예측값 : ", h, "\n 예축값 : ", p, "\n Accuracy : ", a)

