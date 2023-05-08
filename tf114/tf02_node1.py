# 텐서머신 안에 그래프를 만들고 그래프에 값을 넣어줌
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

# node3 = node1 + node2 
node3 = tf.add(node1, node2)       # 위 코드와 결과 동일

print(node1)    # Tensor("Const:0", shape=(), dtype=float32)
print(node2)    # Tensor("Const_1:0", shape=(), dtype=float32)
print(node3)    # Tensor("add:0", shape=(), dtype=float32) 

# sess = tf.Session()
sess = tf.compat.v1.Session() 
print(sess.run(node3))      # 7.0

print(sess.run(node1))


