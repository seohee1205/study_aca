import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
tf.compat.v1.set_random_seed(337)

#1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]   # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                     # (6, 1)

##########################################
# [실습] 시그모이드 빼고 그냥 맹글어봐
##########################################

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))     # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2000
    
for step in range(epochs):
    cost_val, _ , w_val, b_val= sess.run([loss, train, w, b], 
                                feed_dict={x: x_data, y: y_data})
        
    if step % 20 == 0:
        print(epochs, 'loss : ', cost_val)

#4. 평가, 예측            
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
# y_predict = x_data * w_val + b_val    # 넘파이랑 텐서랑 행렬곱했더니 에러남, 그래서 아래 matmul 사용해야 됨
y_predict = tf.matmul(x_test, w_val) + b_val

y_aaa = sess.run(y_predict, feed_dict={x_test:x_data})
print(type(y_aaa))  # <class 'numpy.ndarray'>

r2 = r2_score(y_aaa, y_data)
print('r2 : ', r2)

mse = mean_squared_error(y_aaa, y_data)
print('mse : ', mse)

sess.close()

# r2 :  -5.2328682688071835
# mse :  1.5738497393805584