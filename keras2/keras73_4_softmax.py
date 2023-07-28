import numpy as np
import matplotlib.pyplot as plt

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels, shadow= True, startangle=90)
# 파이 차트를 생성하기 위해 파이썬의 Matplotlib 라이브러리를 사용하는 코드
# ratio: 파이 차트의 각 섹션에 대한 비율을 나타내는 숫자의 리스트
# labels: 파이 차트의 각 섹션에 대한 레이블을 나타내는 문자열의 리스트
# shadow: 파이 차트에 그림자 효과를 줄지 여부를 나타내는 불리언 값. True로 설정하면 그림자 효과가 적용됨
# startangle: 파이 차트의 시작 각도를 나타내는 숫자. 기본값은 0이며, 90으로 설정하면 섹션이 12시 방향에서 시작함

plt.show()