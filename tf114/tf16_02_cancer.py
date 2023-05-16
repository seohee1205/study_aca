import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

tf.compat.v1.set_random_seed(337)

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape)     # (569, 30) (569,)

y = y.reshape(-1, 1)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert the input data to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Define the placeholders for input features and target
x_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Define the variables for weights and bias with dtype=tf.float32
w = tf.compat.v1.Variable(tf.random.normal([30, 1], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

# Define the model
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x_place, w) + b)  # 0에서 1사이의 값

epsilon = 1e-8
# Define the loss function
loss = -tf.reduce_mean(y_place * tf.math.log(hypothesis + epsilon) + (1 - y_place) * tf.math.log(1 - hypothesis + epsilon))

# Define the optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

# Create a session and initialize variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
epochs = 1000
for epoch in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x_place: x_train, y_place: y_train})
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss_val)

# 4. 평가, 예측
x_test_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y_predict = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x_test_place, w) + b)
y_predict = tf.cast(y_predict > 0.5, dtype=tf.float32)

y_aaa = sess.run(y_predict, feed_dict={x_test_place: x_test})


acc = accuracy_score(y_aaa, y_test)
print('acc:', acc)

mse = mean_squared_error(y_aaa, y_test)
print('mse:', mse)

sess.close()
