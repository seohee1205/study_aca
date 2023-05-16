import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
tf.compat.v1.set_random_seed(337)


#1. 데이터
path = './_data/dacon_diabetes/'
train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)
test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)
# print(x_pf.shape)      

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, shuffle= True, train_size=0.8, random_state=1030
)

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.fit_transform(x_test)
# 모델 구성
input_dim = X_train.shape[1]

# 입력 플레이스홀더 정의
x = tf.placeholder(tf.float32, [None, input_dim])
y_true = tf.placeholder(tf.float32, [None, 1]) 

# 가중치와 편향 변수 정의
W = tf.Variable(tf.zeros([input_dim, 1]))
b = tf.Variable(tf.zeros([1]))

# 로지스틱 회귀 모델 정의
logits = tf.matmul(x, W) + b
y_pred = tf.nn.sigmoid(logits)

# 손실 함수 정의 (크로스 엔트로피)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))

# 최적화 함수 정의 (경사 하강법)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 예측 결과 확인
predictions = tf.cast(tf.round(y_pred), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y_true, tf.int32)), tf.float32))

# 텐서플로우 그래프 실행
epochs = 100
batch_size = 32

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        total_batches = X_train.shape[0] // batch_size
        
        for batch in range(total_batches):
            batch_indices = np.random.choice(X_train.shape[0], batch_size)
            batch_x = X_train[batch_indices]
            batch_y = y_train[batch_indices].reshape(-1, 1)  
            
            sess.run(optimizer, feed_dict={x: batch_x, y_true: batch_y})
        
        if epoch % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: X_test, y_true: y_test.reshape(-1, 1)})  
            print("Epoch:", epoch, "Accuracy:", acc)
            
            
# Epoch: 0 Accuracy: 0.9649123
# Epoch: 10 Accuracy: 0.9736842
# Epoch: 20 Accuracy: 0.98245615
# Epoch: 30 Accuracy: 0.98245615
# Epoch: 40 Accuracy: 0.98245615
# Epoch: 50 Accuracy: 0.98245615
# Epoch: 60 Accuracy: 0.98245615
# Epoch: 70 Accuracy: 0.98245615
# Epoch: 80 Accuracy: 0.99122804
# Epoch: 90 Accuracy: 0.99122804