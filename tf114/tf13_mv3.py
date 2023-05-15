import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
tf.compat.v1.set_random_seed(337)

x_data = [[73, 51, 65],                 # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))     # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 2000
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: x_data, y: y_data})
        if step % 10 == 0:
            print(step, loss_val, w_val, b_val)
    
    y_pred = tf.compat.v1.matmul(x, w) + b
    r2 = r2_score(y_data, sess.run(y_pred, feed_dict={x: x_data}))
    print('r2 score:', r2)
    #############################