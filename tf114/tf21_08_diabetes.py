import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

y = y.reshape(-1, 1)        
print(x.shape, y.shape)     # (442, 10) (442, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([10, 15], dtype=tf.float32), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([15], dtype=tf.float32), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random.normal([15, 7], dtype=tf.float32), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([7], dtype=tf.float32), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([7, 10], dtype=tf.float32), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([10], dtype=tf.float32), name='bias3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([10, 13], dtype=tf.float32), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([13], dtype=tf.float32), name='bias4')
layer4 = tf.nn.sigmoid(tf.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([13, 5], dtype=tf.float32), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([5], dtype=tf.float32), name='bias5')
layer5 = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)

w6 = tf.compat.v1.Variable(tf.random.normal([5, 1], dtype=tf.float32), name='weight6')
b6 = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias6')
hypothesis = tf.nn.sigmoid(tf.matmul(layer5, w6) + b6)


# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.009)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train.tolist()})

        if step % 20 == 0:
            print("Step:", step, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = tf.sigmoid(hypothesis)
    y_pred_label = tf.cast(y_pred > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_label, y), dtype=tf.float32))

    y_pred_result, accuracy_result = sess.run([y_pred_label, accuracy], feed_dict={x: x_test, y: y_test})

    print("Predictions:", y_pred_result)
    print("Accuracy:", accuracy_result)
    
    # Accuracy: 0.6315789