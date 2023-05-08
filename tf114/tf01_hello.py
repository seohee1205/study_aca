import tensorflow as tf
print(tf.__version__)

print("hello world")

aaa = tf.constant('hello world')
print(aaa)      # Tensor("Const:0", shape=(), dtype=string)


# sess = tf.Session()     # session을 정의하고 sess 생성  / 1.14 버전
sess = tf.compat.v1.Session()   # 신버전
print(sess.run(aaa))    # b'hello world'


