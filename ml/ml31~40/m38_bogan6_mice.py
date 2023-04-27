# MICE(Multiple Imputation by Chained Equations)

import numpy as np
import pandas as pd
import sklearn as sk
from impyute.imputation.cs import mice


# print(sk.__version__)   # 1.0.2

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


# impute_df = mice(data)   # mice에는 numpy로 넣어줘야함 (안 그러면 에러 뜸)
# AttributeError: 'DataFrame' object has no attribute 'as_matrix'

# impute_df = mice(data.values)       # .values : 넘파이로 바꿀 때 사용
impute_df = mice(data.to_numpy())     # .to_numpy() : 넘파이로 바꿀 때 사용
## 넘파이를 판다스로 바꿀 때: pd.DataFrame() 함수 사용, pd.DataFrame.from_records() 메소드를 사용
## 리스트를 넘파이로 바꿀 때: numpy.array() 함수를 사용
## 리스트를 판다스로 바꿀 때: pd.DataFrame() 함수를 사용


print(impute_df)
