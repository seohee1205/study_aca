# 회귀 데이터 싹 모아서 모델 만들어서 테스트

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
import pandas as pd

path_ddarung = './_data/ddarung/'
path_kaggle_bike = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col = 0)
ddarung_test = pd.read_csv(path_ddarung + 'test.csv', index_col = 0)
kaggle_train = pd.read_csv(path_kaggle_bike + 'train.csv', index_col = 0)  
kaggle_test = pd.read_csv(path_kaggle_bike + 'test.csv', index_col = 0)

ddarung_train = ddarung_train.dropna() # 결측치 제거
ddarung_test = ddarung_test.dropna()

x_d_train = ddarung_train.drop(['count'], axis = 1)
y_d_train = ddarung_train['count']
x_d_pred = ddarung_test

x_k_train = kaggle_train.drop(['count', 'casual', 'registered'], axis = 1)
y_k_train = kaggle_train['count']

data_list= [fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]
model_list= [DecisionTreeRegressor(), RandomForestRegressor()]

for i in range(len(data_list)):
    if i<2:
        x, y= i(return_X_y=True)
    elif i==2:
        x= ddarung_train.drop(['count'], axis =1)
        y = ddarung_train['count']
    elif i==3:
        x= kaggle_train.drop(['count', 'casual', 'registered'], axis = 1)
        y= kaggle_train['count']




#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
# x, y = fetch_california_housing(return_X_y= True)

# print(x.shape, y.shape)         # (150, 4) (150,)

#2. 모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, RandomTreesEmbedding

# model = Sequential()
# model.add(Dense(10, activation= 'relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation= 'softmax'))

# model = LinearSVC(C= 0.3)
# model = LogisticRegressor()
# model = LogisticRegression()        # 분류 / sigmoid
# model = DecisionTreeRegression()
# model = DecisionTreeClassifier()
# model = RandomForestRegressor()


# C는 정규화 매개변수
# C 값이 작을수록 SVM은 분류 오류를 허용하고 결정 경계를 유연하게 만듭니다. 
# 반면, C 값이 커질수록 SVM은 분류 오류를 최소화하고 결정 경계를 보다 엄격하게 만듭니다. 
# 즉, C 값이 커지면 SVM은 학습 데이터에 더욱 적합(fit)하게 되며, 
# 이는 일반화(generalization) 능력을 저해할 수 있습니다.


#3. 컴파일, 훈련
# model.compile(loss= 'sparse_categorical_crossentropy',          # 원핫포함 코드/ 위에서 원핫 안 했을 때 사용, 앞에 0부터 시작하는지 확인해야함 /머신러닝 > 딥러닝
#               optimizer = 'adam',
#               metrics=['acc']) 
# model.fit(x, y, epochs= 100, validation_split= 0.2)

# model.fit(x, y)


#4. 평가, 예측
# results = model.evaluate(x, y)

# results = model.score(x, y)

# print(results)      # 0.9666666666666667

# 어떤 모델을 사용할 건지 / 모델 돌아가는 구성 방식 보기, 
     
          

