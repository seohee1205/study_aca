import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

# 항상 변수를 초기화 해줘야 한다.
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(x + y))



