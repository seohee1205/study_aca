import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_covtype
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]
print(train_csv.shape) #(5497,13)
 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)   #[1 0 1 ... 1 1 1]
print(type(aaa))  #<class 'numpy.ndarray'>
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white'])) #[0 1]


#1-1 결측치 제거 
# print(train_csv.isnull().sum()) #결측치없음 

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [3 4 5 6 7 8 9]
y = y-3
# print(type(y))
# print(y)
print("y_shape:", y.shape)           #(5497,) 
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2 3 4 5 6]
# test_csv = test_csv.drop(['type'], axis=1)

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


# dacon_wine
# model.score :  0.649090909090909
# Voting.acc :  0.649090909090909