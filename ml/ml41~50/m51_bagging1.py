import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 123, train_size= 0.8, shuffle= True, stratify=y 
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators= 10,     # 모델을 10번 돌린다
                          n_jobs= -1,
                          random_state= 337,
                        #   bootstrap= True,
                          bootstrap = False
                          )
# DecisionTree를 10번 배깅한다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))

# DecisionTree 결과
# model.score :  0.9649122807017544
# acc :  0.9649122807017544

# RandomForest 결과
# model.score :  0.9824561403508771
# acc :  0.9824561403508771

# Bagging 에 10번 돌린 결과
# model.score :  0.9912280701754386
# acc :  0.9912280701754386