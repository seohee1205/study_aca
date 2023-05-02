import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_absolute_error

x, y, = load_linnerud(return_X_y=True)
# print(x)
# print(y)
print(x.shape, y.shape)     # (20, 3) (20, 3)
########################################################################
# 원래 : [2, 110, 43]   -> 예상 : [138. 33. 68.]
########################################################################
# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred), 4))   # Ridge 스코어 :  7.4569
# print(model.predict([[2, 110, 43]]))    # [[187.32842123  37.0873515   55.40215097]]

# model = XGBRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred), 4))   # XGBRegressor 스코어 :  0.0008
# print(model.predict([[2, 110, 43]]))    # [[138.00215   33.001656  67.99831 ]]

# model = LGBMRegressor()         # 에러
# model.fit(x, y)
# print("스코어 : ", model.score(x, y))   
# print(model.predict([[2, 110, 43]]))    
# # ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# # # 훈련을 3번 해야함 (각각 훈련한 후 컨켓) or MultiOutput

# model = MultiOutputRegressor(LGBMRegressor()) 
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred), 4))   # MultiOutputRegressor 스코어 :  8.91
# print(model.predict([[2, 110, 43]]))    # [[178.6  35.4  56.1]]

# model = CatBoostRegressor()     # 에러
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred), 4))   # MultiOutputRegressor 스코어 :  8.91
# print(model.predict([[2, 110, 43]]))    # [[178.6  35.4  56.1]]

# model = MultiOutputRegressor(CatBoostRegressor()) 
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred), 4))   # MultiOutputRegressor 스코어 :  0.2154
# print(model.predict([[2, 110, 43]]))    # [[138.97756017  33.09066774  67.61547996]]


model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어 : ",
      round(mean_absolute_error(y, y_pred), 4))   # CatBoostRegressor 스코어 :  0.0638
print(model.predict([[2, 110, 43]]))    # [[138.21649371  32.99740595  67.8741709 ]]


