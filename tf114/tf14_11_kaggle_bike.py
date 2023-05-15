import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
if  tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# 1. 데이터
path = './_data/kaggle_bike/'

# Load the diabetes dataset
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# Reshape y to have shape (n_samples, 1)
train_csv = train_csv.dropna()
x = train_csv.drop(['count','casual','registered'], axis=1)
y = train_csv['count'].values
y = y.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=337)

# Convert the input data to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Define the placeholders for input features and target
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Define the variables for weights and bias with dtype=tf.float32
w = tf.compat.v1.Variable(tf.random.normal([8, 1], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

# Define the model
hypothesis = tf.compat.v1.matmul(x, w) + b

# Define the loss function
loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y))

# Define the optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# Create a session and initialize variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
epochs = 1000
for epoch in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: X_train, y: y_train})
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss_val)

# Evaluation
y_train_pred = sess.run(hypothesis, feed_dict={x: X_train})
y_test_pred = sess.run(hypothesis, feed_dict={x: X_test})

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R2 Score (Train):", r2_train)
print("R2 Score (Test):", r2_test)

# Close the session
sess.close()