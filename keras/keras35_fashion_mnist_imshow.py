from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# print(x_train)
# print(y_train)
print(x_train[1500])
print(y_train[12000])

import matplotlib.pyplot as plt
plt.imshow(x_train[12000])
plt.show()

