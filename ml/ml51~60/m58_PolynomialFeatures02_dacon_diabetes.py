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
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
path = './_data/dacon_diabetes/'
train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)
test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)
# print(x_pf.shape)      

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


# dacon_diabetes
# model.score :  0.7709923664122137
# Voting.acc :  0.7709923664122137
