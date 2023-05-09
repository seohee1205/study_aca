import tensorflow as tf
tf.compat.v1.set_random_seed(337)

# 1. 데이터
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
x_data = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)       # Variable: 변수 초기화
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. 모델 구성
hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08235)
train = optimizer.minimize(loss)  # loss의 최소값을 뽑는다

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 전체가 초기화됨

    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:[1, 2, 3, 4, 5], y:[2, 4, 6, 8, 10]})
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)  # verbose

    # 훈련된 모델을 이용하여 x_data에 대한 예측값을 구한다
    x_data = [6, 7, 8]
    y_pred = sess.run(hypothesis, feed_dict={x: x_data})

    # # 예측값 출력
    print(y_pred)



# [실습]
# 08_2를 카피해서 아래를 맹그러봐

################## 1. Session() // sess.run(변수)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(w)
print('aaa : ', aaa)    # aaa :  [-1.5080816   0.26086742]
sess.close()

################## 2. Session() // 변수.eval(Session=sess)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = w.eval(session=sess)   # 텐서플로 데이터형인 '변수'를 파이썬에서 볼 수 있는 놈으로 바꿔줘
print('bbb : ', bbb)    # bbb :  [-1.5080816   0.26086742]
sess.close()

################## 3. InteractiveSession() // 변수.eval
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = w.eval()
print('ccc : ', ccc)    # ccc :  [-1.5080816   0.26086742]
sess.close()

# aaa :  [-1.5080816   0.26086742]
# bbb :  [-1.5080816   0.26086742]
# ccc :  [-1.5080816   0.26086742]
