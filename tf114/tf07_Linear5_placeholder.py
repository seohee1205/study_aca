import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = tf.placeholder(tf.float32, shape = [None])
y = tf.placeholder(tf.float32, shape = [None])

w = tf.Variable(tf.random_normal([1]), dtype= tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype= tf.float32)

#2. 모델 구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(loss)    # loss의 최소값을 뽑는다

#3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())     # 전체가 초기화됨

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:[1, 2, 3, 4, 5], y:[2, 4, 6, 8, 10]})
        if step %20 == 0:
            print(step, loss_val, w_val, b_val)   # verbose


