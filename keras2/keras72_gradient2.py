import numpy as np
import matplotlib.pyplot as  plt

f = lambda x: x**2 -4*x +6

gradient = lambda x : 2*x -4 

x = -10
epochs = 30
learning_rate = 0.25

x_history = []  # x 값 저장
f_history = []  # f(x) 값 저장

print("epoch\t x\t f(X)")
print("{:02d}\t {:6.5f}\t {6.5f}\t".format(0, x, f(x)))

for i in range(epochs):
    x_history.append(x)
    f_history.append(f(x))
    x = x - learning_rate * gradient(x)
    
    # print(i, x, f(x))
    
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))
    
    
# 그래프 그리기
plt.plot(x_history, f_history, 'ro-')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')
plt.grid(True)
plt.show()