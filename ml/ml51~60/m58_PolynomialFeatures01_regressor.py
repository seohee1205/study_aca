import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터
# x, y  = load_diabetes(return_X_y=True)
x, y  = fetch_california_housing(return_X_y=True)


pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)
print(x_pf.shape)       # (150, 15)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, shuffle= True, train_size=0.8, random_state=1030
)

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.fit_transform(x_test)


#2. 모델
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print("Voting.r2 : ", r2_score(y_test, y_pred))


# load_diabetes
# model.score :  0.4457965716134076
# Voting.r2 :  0.4457965716134076

# fetch_california_housing
# model.score :  0.43416523124734996
# Voting.r2 :  0.43416523124734996