import tensorflow  as tf
tf.set_random_seed(337)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

w = tf.Variable(333, dtype= tf.float32)
b = tf.Variable(111, dtype= tf.float32)

# [실습] 맹그러

#2. 모델 구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0156)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 9999
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose
        
sess.close()    # 수동옵션 저장되는 것을 방지

