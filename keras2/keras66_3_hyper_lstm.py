# 1, 2, 3 파일 모두 공통적용
# early_stopping 적용
# MCP 적용
# 레이어 자체는 적용 안 했잖아 -> 레이어 적용

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float32') / 255.

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, lr=0.001):
    model = Sequential()
    model.add(LSTM(node1, input_shape=(28, 28), activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model


def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam(learning_rates)', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    nodes = [32, 64, 128, 256]
    
    return {
        'batch_size': batchs,
        'optimizer': optimizers,
        'drop': dropouts,
        'activation': activations,
        'lr': learning_rates,
        'node1': nodes,
        'node2': nodes,
    }

hyperparameters = create_hyperparameter()

keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

filepath = './_save/MCP/keras66_2/'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='acc', patience=10, mode='max',
                   verbose=1, restore_best_weights=True)

mcp = ModelCheckpoint(monitor='acc', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath + 'k27_{epoch:04d}-{loss:.4f}.hdf5')

import time
start = time.time()
model.fit(x_train, y_train, epochs=25, callbacks=[es, mcp])
end = time.time()

print("걸린시간 : " , end -start)
print("Best Score: ", model.best_score_) # 훈련 데이터에서 최고 점수
print("Best Params: ", model.best_params_)
print("model Score: ", model.score(x_test, y_test)) # 테스트 데이터의 점수


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_predict))

# model Score:  0.9879000186920166
# Accuracy: 0.9879  