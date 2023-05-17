import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

tf.set_random_seed(4145)

# 1 데이터
random_state = 12351356

ddarung_path = './_data/_dacon_ddarung/'
kaggle_bike_path = './_data/_kaggle_bike/'

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
    try:
        if d < 2:
            x, y = data_list[d](return_X_y=True)
            y = y.reshape(-1, 1)
        else:
            x, y = data_list[d]
            y = y.values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = random_state, shuffle = True)
        
        scaler = StandardScaler()
        
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        x_train = x_train.astype(np.float64)
        x_test = x_test.astype(np.float64)
        
        n_features = x_train.shape[1]
        
        n_neurons1 = 10
        n_neurons2 = 10
        
        x_p = tf.compat.v1.placeholder(tf.float32, shape = [None, n_features])
        y_p = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

        w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, n_neurons1], name = 'weight1'))
        b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons1], name = 'bias1'))
        layer1 = tf.compat.v1.matmul(x_p, w1) + b1
        
        w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight2'))
        b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons2], name = 'bias2'))
        layer2 = tf.compat.v1.matmul(layer1, w2) + b2
        
        w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight3'))
        b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons2], name = 'bias3'))
        layer3 = tf.compat.v1.matmul(layer2, w3) + b3
        
        w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight4'))
        b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons2], name = 'bias4'))
        layer4 = tf.compat.v1.matmul(layer3, w4) + b4
        
        w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons1, n_neurons2], name = 'weight5'))
        b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([n_neurons2], name = 'bias5'))
        layer5 = tf.compat.v1.matmul(layer4, w5) + b5
        
        w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_neurons2, 1], name = 'weight6'))
        b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias6'))        
        
        # hypothesis = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(layer5, w6) + b6)
        hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer5, w6) + b6) # 둘다됨
        
        # 3-1 컴파일
        loss = tf.reduce_mean(tf.square(hypothesis - y_p))
        
        train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.0000001).minimize(loss)

        # 3-2 훈련
        sess = tf.compat.v1.Session()

        sess.run(tf.compat.v1.global_variables_initializer())

        epochs = 101
        
        for s in range(epochs):
            _, loss_val = sess.run([train, loss], feed_dict = {x_p : x_train, y_p : y_train})
            
            if s % 20 == 0:  # Print loss every 200 steps
                print(f'step : {s}, loss : {loss_val}')

            y_predict = sess.run(hypothesis, feed_dict = {x_p : x_test})

        # 4 평가
        r2 = r2_score(y_test, y_predict)
        print(f'데이터 : {d}, r2_score : {r2}')
    except ValueError as ve:
        print(f'데이터 : {d}, 에러다 ㅋㅋ : {ve}') # 어떤 데이터셋에서 에러가 발생하는지 확인