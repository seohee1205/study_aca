import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__)   # 1.0.2

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer  # 결측치에 대한 책임을 돌릴 것 같아
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# imputer = SimpleImputer()       # 디폴트 : 평균
# imputer = SimpleImputer(strategy='mean')       # 평균
# imputer = SimpleImputer(strategy='median')       # 중위값
# imputer = SimpleImputer(strategy='most_frequent')       # 최빈값 // 개수가 같을 경우 가장 작은 값
# imputer = SimpleImputer(strategy='constant')       # 0 들어감
# imputer = SimpleImputer(strategy='constant', fill_value= 7777)       # 7777 들어감
# imputer = KNNImputer()

# imputer = IterativeImputer()
# imputer = IterativeImputer(estimator = DecisionTreeRegressor())
imputer = IterativeImputer(estimator = XGBRegressor())    


data2 = imputer.fit_transform(data)
print(data2)



