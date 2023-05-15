import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from sklearn.metrics import r2_score

#1. 데이터
#          첫번 두번 세번 네번 다섯번
x1_data = [73., 93., 89., 96., 73.]         # 국어 
x2_data = [80., 88., 91., 98., 66.]         # 영어
x3_data = [75., 93., 90., 100., 70.]        # 수학
y_data = [152., 185., 180., 196., 142.]     # 환산점수

# [실습] 맹그러봐

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)   # w 3개, b 1개
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
 
#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))     # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# r2 스코어까지 만들기

#3-2. 훈련

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

# #3-2. 훈련

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     epochs = 100
#     for step in range(epochs):
#         _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b],
#                                                               feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
#         if step % 20 == 0:
#             print(step, loss_val, w1_val, w2_val, w3_val, b_val)

    epochs = 2001
    for step in range(epochs):
        # cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
        cost_val, _ = sess.run([loss, train],                               
                                        feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
        if step % 20 ==0:
            print(epochs, 'loss : ', cost_val)

# #4. 평가
#     x_test_data = [93., 88., 89., 90., 70.]
#     y_test_data = [185., 180., 182., 179., 150.]
    
#     y_predict_val = sess.run(hypothesis, feed_dict={x1: x_test_data, x2: x_test_data, x3: x_test_data, b: b_val})
#     r2 = r2_score(y_test_data, y_predict_val)

#     print("R2 Score:", r2)   

# R2 Score: 0.8682793069880151
