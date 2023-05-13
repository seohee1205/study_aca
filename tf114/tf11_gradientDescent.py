import tensorflow as tf

x_train = [1, 2, 3]     # [1]
y_train = [1, 2, 3]     # [2]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype= tf.float32, name = 'weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

############### optimizer ###############
lr = 0.1
# gradient = tf.reduce_mean((w * x - y) * x)    # 이렇게 쓰면 안 돼
gradient = tf.reduce_mean((x * w - y) * x)      # gradient의 식(loss의 미분값)
# gradient = tf.reduce_mean((hypothesis - y) * x)

descent = w - lr * gradient
update = w.assign(descent)      # w = w - lr * gradient -> 기울기 값을 계속 업데이트

##################### 옵티마이저 #####################
w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
    
sess.close()
print('=============== w history ===============')
print(w_history)
print('=============== Loss history ===============')
print(loss_history)


# 체인룰: 미분에 미분 = 미분미분
