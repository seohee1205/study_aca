##### 현재 버전이 1.0이면 그냥 출력 //버전도 print 해줘
##### 현재 버전이 2.0이면 즉시실행모드를 끄고 출력
##### if문 써서 1번 소스를 변경

import tensorflow as tf
print(tf.__version__)

'''
# 즉시 실행모드
print(tf.executing_eagerly())   # True
tf.compat.v1.disable_eager_execution()      # 즉시 실행모드 끔 / 텐서 2.0을 1.0 방식으로 
print(tf.executing_eagerly())   # False

tf.compat.v1.enable_eager_execution()      # 즉시 실행모드 킴 / 텐서 1.0을 2.0 방식으로 
print(tf.executing_eagerly())   # True

aaa = tf.constant('hello world')
sess = tf.compat.v1.Session()
# print(sess.run(aaa))    # tf2.0에서는 없어짐
'''


if tf.__version__ >= '2.0':
    tf.compat.v1.disable_eager_execution()
    print(tf.executing_eagerly())
else:
    tf.compat.v1.enable_eager_execution() 
    print(tf.executing_eagerly())

    
