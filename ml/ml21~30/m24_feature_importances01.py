import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


#1. 데이터
# x, y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size = 0.8, shuffle=True, random_state= 337 
# )

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# [실습] for문 써서 4개 돌리기
data_list = [load_iris, load_breast_cancer, load_digits, load_wine]
model_list = [GradientBoostingClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), XGBClassifier()]

for i in data_list:
    x, y = i(return_X_y=True)
    x_train, x_test, y_train, y_test= train_test_split(
        x, y, shuffle= True, train_size= 0.8, random_state= 337
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #2. 모델
    for j in model_list:
        model = j
        #3. 컴파일, 훈련
        model.fit(x_train, y_train)

        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(i.__name__, type(j).__name__, 'results : ', results)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(i.__name__, type(j).__name__, 'acc :', acc)

        print(i.__name__, type(j).__name__, ':', model.feature_importances_)

        



#2. 모델
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()


#3. 훈련
# model.fit(x_train, y_train)

#4. 평가, 예측
# result = model.score(x_test, y_test)
# print("model.score : ", result)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score : ', acc)

# print("==================================")
# print(model, ":", model.feature_importances_)
# accuracy_score :  0.9666666666666667
# DecisionTreeClassifier() : [0.01671193 0.         0.93062443 0.05266364]
# accuracy_score :  0.9666666666666667
# RandomForestClassifier() : [0.11808564 0.03303654 0.44094527 0.40793256]
# accuracy_score :  0.9666666666666667
# GradientBoostingClassifier() : [0.00596332 0.01349118 0.70018421 0.28036129]
# accuracy_score :  0.9666666666666667
# XGBClassifier : [0.01794496 0.01218657 0.8486943  0.12117416]

# accuracy_score :  0.9666666666666667
