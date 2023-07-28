# 난 정말 시그모이드 ~~ ♪♬

import numpy as np
import matplotlib.pyplot as plt

# def sifmoid(x):
#     return 1 / (1 + np.exp(-x))

sigmoid = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()
