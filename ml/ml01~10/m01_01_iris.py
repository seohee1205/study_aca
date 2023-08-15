### 분류 데이터들 싹 모아서 테스트

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, fetch_covtype, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings(action = 'ignore')

index1 = [load_iris, load_breast_cancer, load_digits, fetch_covtype, load_wine]
index2 = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

scaler = MinMaxScaler() #

for i in index1:
    x, y = i(return_X_y=True)
    x = scaler.fit_transform(x)
    for j in index2:
        model = j
        model.fit(x, y)
        results = model.score(x, y)
        print(i.__name__, type(j).__name__,results)
        

#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
# x, y = load_iris(return_X_y= True)

# print(x.shape, y.shape)         # (150, 4) (150,)

#2. 모델 구성

# model = Sequential()
# model.add(Dense(10, activation= 'relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation= 'softmax'))


# model = LinearSVC(C= 0.3)
# model = LogisticRegression()       # 분류(회귀)
## model = DecisionTreeRegression()
# model = DecisionTreeClassifier()   # 분류
## model = RandomForestRegressor()
# model = RandomForestClassifier()   # 분류


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

# print(results)      

# 어떤 모델을 사용할 건지 / 모델 돌아가는 구성 방식 보기, 


# load_iris LinearSVC 0.9466666666666667
# load_iris LogisticRegression 0.94
# load_iris DecisionTreeClassifier 1.0
# load_iris RandomForestClassifier 1.0
# load_breast_cancer LinearSVC 0.9806678383128296
# load_breast_cancer LogisticRegression 0.9718804920913884
# load_breast_cancer DecisionTreeClassifier 1.0
# load_breast_cancer RandomForestClassifier 1.0
# load_digits LinearSVC 0.9888703394546466
# load_digits LogisticRegression 0.9844184752365053
# load_digits DecisionTreeClassifier 1.0
# load_digits RandomForestClassifier 1.0
# fetch_covtype LinearSVC 0.7126634217537675
# fetch_covtype LogisticRegression 0.7200161098221723
# fetch_covtype DecisionTreeClassifier 1.0
# fetch_covtype RandomForestClassifier 1.0
# load_wine LinearSVC 0.9943820224719101
# load_wine LogisticRegression 0.9887640449438202
# load_wine DecisionTreeClassifier 1.0
# load_wine RandomForestClassifier 1.0
          

