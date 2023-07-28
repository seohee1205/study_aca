import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0, x)

relu = lambda x: np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# [실습]
#3_2, 3_3, 3_4 ...
# relu, selu, reaky_relu, ThresholdedReLU layer, PReLU layer...

import numpy as np
import matplotlib.pyplot as plt

# ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# ELU (Exponential Linear Unit)
def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# SELU (Scaled Exponential Linear Unit)
def selu(x, alpha=1.67326, scale=1.0507):
    condition = x > 0
    return scale * np.where(condition, x, alpha * (np.exp(x) - 1))

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# Thresholded ReLU
def thresholded_relu(x, threshold=1.0):
    return np.where(x > threshold, x, 0)

# PReLU (Parametric Rectified Linear Unit)
def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

x = np.arange(-5, 5, 0.1)

# Calculate y values for each function
y_relu = relu(x)
y_elu = elu(x)
y_selu = selu(x)
y_leaky_relu = leaky_relu(x)
y_thresholded_relu = thresholded_relu(x)
y_prelu = prelu(x)

# Plotting
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_elu, label='ELU')
plt.plot(x, y_selu, label='SELU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_thresholded_relu, label='Thresholded ReLU')
plt.plot(x, y_prelu, label='PReLU')
plt.legend()
plt.grid()
plt.show()
