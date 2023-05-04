# 실습
# 맹그러 10개

# VIF 할 떈 먼저 스케일링 한 후 y 넣지 않는다??

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

#1. 데이터셋
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns= datasets.feature_names)
df['target'] = datasets.target
# print(df)

y = df['target']
x = df.drop(['target'], axis= 1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 다중공선성
vif = pd.DataFrame()
vif['variables'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
print(vif)
#            variables        VIF
# 0  sepal length (cm)   7.072722
# 1   sepal width (cm)   2.100872
# 2  petal length (cm)  31.261498
# 3   petal width (cm)  16.090175

x = x.drop(['petal length (cm)'], axis= 1)     # 제거 후 결과 :  0.9029635316698656
print(x)

############ petal length (cm) 제거 후
x_scaled = df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]

vif2 = pd.DataFrame()
vif2['variables'] = x.columns

vif2['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
print(vif2)
#            variables        VIF
# 0  sepal length (cm)  94.373039
# 1   sepal width (cm)  52.984682
# 2   petal width (cm)  11.868708


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state= 337, test_size= 0.2, # stratify=y
)

scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

#2. 모델
model = RandomForestRegressor(random_state= 337)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

# 결과 :  0.950514395393474
# 'petal length (cm)' 제거 후 결과 :  0.9029635316698656


