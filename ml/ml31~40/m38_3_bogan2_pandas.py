import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0. 결측치 확인
print(data.isnull())        # True가 결측치임
print(data.isnull().sum())  
print(data.info())

#1. 결측치 삭제
print("================== 결측치 삭제 ==================")
# print(data['x1'].dropna())  # 이렇게 하면 그 열에서만 삭제돼서 그게 그거임
print(data.dropna())        # 디폴트가 행 위주로 삭제
print("================== 결측치 삭제 ==================")
print(data.dropna(axis=0))  # 행위주 삭제 / 디폴트     //  0: 행, 1: 열
print("================== 결측치 삭제 ==================")
print(data.dropna(axis=1))  # 열위주 삭제 / 디폴트 

#2-1. 특정값 - 평균
print("================== 결측치 처리 mean() ==================")
means = data.mean()
print('평균 : ', means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
print("================== 결측치 처리 median() ==================")
median = data.median()
print('중위값 : ', median)
data3 = data.fillna(median)
print(data3)

#2-3. 특정값 - ffill, bfill
print("================== 결측치 처리 ffill, bfill ==================")
data4 = data.fillna(method='ffill')
print(data4)
data5 = data.fillna(method='bfill')
print(data5)

#2-4. 특정값 - 임의값으로 채우기
print("================== 결측치 처리 - 임의의 값으로 채우기 ==================")
# data6 = data.fillna(7777777)
data6 = data.fillna(value=7777777)  # 위와 동일
print(data6)


######################### 특정 컬럼만 !!! #########################

#1. x1 컬럼에 평균값을 넣고
means = data['x1'].mean()
data['x1'] = data['x1'].fillna(means)
print(data)

#2. x2 컬럼에 중위값을 넣고
medians = data['x2'].median()
data['x2'] = data['x2'].fillna(medians)
print(data)

#3. x4 컬럼에 ffill한 후 / 제일 위에 남은 행에 777777로 채우기
data['x4'] = data['x4'].fillna(method='ffill').fillna(value= 7777777)
print(data)

