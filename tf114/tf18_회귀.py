import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1 데이터
tf.compat.v1.set_random_seed(1234)

random_state = 337

ddarung_path = './_data/dacon_ddarung/'
kaggle_bike_path = './_data/kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1)
y1 = ddarung['count']

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y2 = kaggle_bike['count']

data_list = [load_diabetes,
             fetch_california_housing,
             (x1, y1),
             (x2, y2)]

for d in range(len(data_list)):
    if d < 2:
        x, y = data_list[d](return_X_y = True)
        y = y.reshape(-1, 1) # (442, 1)
    else:
        x, y = data_list[d]
        y = y.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = random_state, shuffle = True)
    n_features = x_train.shape[1]
    
    x_p = tf.compat.v1.placeholder(tf.float32, shape = [None, n_features])
    y_p = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

    w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1], name = 'weight'))
    b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))

    # 2 모델
    logits = tf.matmul(x_p, w) + b
    hypothesis = tf.nn.softmax(logits)

    # 3-1 컴파일
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_p))
   
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

    train = optimizer.minimize(loss)

    # 3-2 훈련
    sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 101
    
    for s in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict = {x_p : x_train, y_p : y_train})
        
        if s % 20 == 0:  # Print loss every 200 steps
            print(f'step : {s}, loss : {loss}')

        y_predict = sess.run(hypothesis, feed_dict = {x_p : x_test})

    # 4 평가
    r2 = r2_score(y_test, y_predict)
    print(f'데이터 : {d}, r2_score : {r2}')
            