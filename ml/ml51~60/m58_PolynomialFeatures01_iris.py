import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_covtype
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
# x, y  = load_iris(return_X_y=True)
# x, y  = load_wine(return_X_y=True)
# x, y  = load_breast_cancer(return_X_y=True)
# x, y  = load_digits(return_X_y=True)
x, y  = fetch_covtype(return_X_y=True)


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
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print("Voting.acc : ", accuracy_score(y_test, y_pred))


# iris
# model.score :  0.9333333333333333
# Voting.acc :  0.9333333333333333

# wine
# model.score :  0.9722222222222222
# Voting.acc :  0.9722222222222222

# load_breast_cancer
# model.score :  0.956140350877193
# Voting.acc :  0.956140350877193

# load_digits
# model.score :  0.9666666666666667
# Voting.acc :  0.9666666666666667

# fetch_covtype
