import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# One-hot 인코딩
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

y_train_labels = np.argmax(y_train_encoded, axis=1)
y_test_labels = np.argmax(y_test_encoded, axis=1)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
w = tf.Variable(tf.random_normal([64, 10]), name='weight')
b = tf.Variable(tf.zeros([1, 10]), name='bias')
y = tf.compat.v1.placeholder(tf.int64, shape=[None])

logits = tf.compat.v1.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train_labels})
        if step % 100 == 0:
            print("Step:", step, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = sess.run(hypothesis, feed_dict={x: x_test})
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print("Accuracy:", accuracy)
    
# Accuracy: 0.15555555555555556