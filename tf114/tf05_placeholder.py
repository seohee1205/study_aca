import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())   # True

# 즉시실행모드
tf.compat.v1.disable_eager_execution()  # 꺼

print(tf.executing_eagerly())   # False

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
# placeholder: 어떤 공간에 값을 받을 준비를 하는 것

add_node = a + b

print(sess.run(add_node, feed_dict= {a:3, b:4.5}))      # 7.5
print(sess.run(add_node, feed_dict= {a:[1,3], b:[2,4]}))    # [3. 7.]

add_and_triple = add_node * 3
print(add_and_triple)       # Tensor("mul:0", dtype=float32)

print(sess.run(add_and_triple, feed_dict= {a:7, b:3}))  # 30.0



