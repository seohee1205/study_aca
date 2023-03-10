import numpy as np

a = np.array([[1, 2, 3], [6, 4, 5], [7, 9, 2], [3, 2, 1], [2, 3, 1]])
print(a)
print(a.shape)      # (5, 3)
print(np.argmax(a))     # 7 : 7번째 위치가 가장 높다 (값이 아니라 위치를 알려준 것)
print(np.argmax(a, axis= 0))    # [2 2 1] : 0은 행이다. 그래서 행끼리 비교한다.     <세로로 비교>
print(np.argmax(a, axis= 1))    # [2 0 1 0 1] : 1은 열이다. 그래서 열끼리 비교한다. <가로로 비교> 
print(np.argmax(a, axis= -1))    # [2 0 1 0 1] : -1은 가장 마지막이라는 뜻이다.
            #  즉 가장 마지막 축이라는 것. 이건 2차원이니까 가장 마지막 축은 1,
            #  그래서 -1을 쓰면 이 데이터는 axis = 1과 동일
            

             
