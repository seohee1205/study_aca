# 회귀 데이터 싹 모아서 모델 만들어서 테스트



import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing,

#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x, y = fetch_california_housing(return_X_y= True)

print(x.shape, y.shape)         # (150, 4) (150,)

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
model = LogisticRegression()        # 분류 / sigmoid
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

model.fit(x, y)


#4. 평가, 예측
# results = model.evaluate(x, y)

results = model.score(x, y)

print(results)      # 0.9666666666666667

# 어떤 모델을 사용할 건지 / 모델 돌아가는 구성 방식 보기, 
     
          

