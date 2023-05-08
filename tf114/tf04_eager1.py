import tensorflow as tf
print(tf.__version__)

# 즉시 실행모드
print(tf.executing_eagerly())   # True
tf.compat.v1.disable_eager_execution()      # 즉시 실행모드 끔 / 텐서 2.0을 1.0 방식으로 
print(tf.executing_eagerly())   # False

tf.compat.v1.enable_eager_execution()      # 즉시 실행모드 킴 / 텐서 1.0을 2.0 방식으로 
print(tf.executing_eagerly())   # True

aaa = tf.constant('hello world')
sess = tf.compat.v1.Session()
# print(sess.run(aaa))    # tf2.0에서는 없어짐


