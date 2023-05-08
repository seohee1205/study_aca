import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# w = tf.Variable(111, dtype= tf.float32)
# b = tf.Variable(100, dtype= tf.float32)   # b 초기값은 항상 0
w = tf.Variable(tf.random_normal([1]), dtype= tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype= tf.float32)

# w = tf.random_normal([1])
# b = tf.random_normal([1])

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w))  # [-0.4121612]

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(w))  # [-0.4121612]


#2. 모델 구성
# y = wx + b
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(loss)    # loss의 최소값을 뽑는다
# model.compile(loss='mse', optimizer='sgd')  # sgd: 확률적 GradientDescent
# w = w - lr * loss를 w로 미분

#3-2. 훈련
with tf.compat.v1.Session() as sess:
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())     # 전체가 초기화됨

    epochs = 2001
    for step in range(epochs):
        sess.run(train)
        if step %20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose
            
    # sess.close()    # 수동옵션 저장되는 것을 방지
    


