# 1, 2, 3 파일 모두 공통적용
# early_stopping 적용
# MCP 적용
# 레이어 자체는 적용 안 했잖아 -> 레이어 적용

# CNN으로 맹그러

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import cross_val_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000, -1).astype('float32')/255.
# x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')/255.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.


# #2. 모델
# def build_model(drop=0.5, optimizer= 'adam',activation='relu', 
#                 node1=64, node2=64, node3=64, lr=0.001):
#     inputs = Input(shape=(784), name= 'inputs')
#     x = Dense(node1, activation=activation, name = 'hidden1')(inputs)
#     x = Dropout(drop)(x)
#     x = Dense(node1, activation=activation, name = 'hidden2')(x)
#     x = Dropout(drop)(x)
#     x = Dense(node1, activation=activation, name = 'hidden3')(x)
#     x = Dropout(drop)(x)    
#     x = Dense(256, activation=activation, name = 'hidden4')(x)
#     x = Dropout(drop)(x)
#     outputs = Dense(10, activation='softmax', name = 'outputs')(x)

#     model = Model(inputs = inputs, outputs = outputs)
    
#     model.compile(optimizer=optimizer, metrics=['acc'],
#                   loss = 'sparse_categorical_crossentropy')
#     return model


# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(28, 28, 1), name='inputs')
    x = Conv2D(32, (3, 3), activation=activation, padding='same', name='conv1')(inputs)
    x = MaxPool2D(pool_size=(2, 2), name='pool1')(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), activation=activation, padding='same', name='conv2')(x)
    x = MaxPool2D(pool_size=(2, 2), name='pool2')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(node1, activation=activation, name='hidden1')(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model


def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
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
        # 'node': nodes
        'node1': nodes,
        'node2': nodes,
        'node3': nodes
    }

hyperparameters = create_hyperparameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

keras_model = KerasClassifier(build_fn = build_model, verbose=1)
# model = GridSearchCV(keras_moodel,hyperparameters, cv =3)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

import datetime
date = datetime.datetime.now()
print(date)  # 2023-03-14 11:11:30.046663
date = date.strftime("%m%d_%H%M") # 시간을 문자로 (월, 일, 시간, 분)
print(date)

filepath = './_save/MCP/keras66_2/'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min',
                   verbose = 1, restore_best_weights = True)

mcp = ModelCheckpoint(monitor='loss', mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath + 'k27_' + date + '_{epoch:04d}-{loss:.4f}.hdf5')

import time
start = time.time()
model.fit(x_train, y_train, epochs=30, callbacks = [es, mcp])
end = time.time()

print("걸린시간 : " ,end -start)
print("Best Score: ", model.best_score_) #train데이터의 최고의 스코어
print("Best Params: ", model.best_params_)
print("Best estimator", model.best_estimator_)
print("model Score: ", model.score(x_test,y_test)) #test의 최고 스코어

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc :', accuracy_score(y_test,y_predict)) #model 스코어랑 같음.


# model Score:  0.992900013923645
# acc : 0.9929