import numpy as np
import pandas as pd
from datetime import datetime

dates = ['4/25/2023', '4/26/2023', '4/27/2023', '4/28/2023', '4/29/2023', '4/30/2023']

dates = pd.to_datetime(dates)
print(dates)
print(type(dates))      # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

print("=============================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index=dates)      # 벡터가 모이면 시리즈 / 벡터와 매치가 됨 = 1차원, 컬럼(열) 1개랑 매치, 하나로 쭉 이어진 데이터 형태
                                                        # 시리즈가 다 모이면 데이터프레임이 됨
print(ts)

print("==============================")
ts = ts.interpolate()
print(ts)


# =============================
# 2023-04-25     2.0
# 2023-04-26     NaN
# 2023-04-27     NaN
# 2023-04-28     8.0
# 2023-04-29    10.0
# 2023-04-30     NaN
# dtype: float64
# ==============================
# 2023-04-25     2.0
# 2023-04-26     4.0
# 2023-04-27     6.0
# 2023-04-28     8.0
# 2023-04-29    10.0
# 2023-04-30    10.0
# dtype: float64

