# [실습] 각종 모델 10개 넣어서 확인해볼 것!

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

'''
#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 123, train_size= 0.8, shuffle= True, stratify=y 
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
aaa = LogisticRegression()
model = BaggingClassifier(aaa,
                          n_estimators= 20,     # 모델을 10번 돌린다
                          n_jobs= -1,
                          random_state= 337,
                          bootstrap= True,    # 디폴트: True
                        #   bootstrap = False
                          )
# DecisionTree를 10번 배깅한다
##### bootstrap = True 쓰면 성능 좋아짐


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))
'''


###########

data_list = [load_iris(return_X_y= True), load_breast_cancer(return_X_y= True), 
            load_digits(return_X_y=True), load_wine(return_X_y= True), fetch_covtype(return_X_y=True)]
data_name = ["아이리스", "캔서", "디지트", "와인", "패치콥타입"]

model_list = [DecisionTreeClassifier(), RandomForestClassifier(), BaggingClassifier()]
model_name = ["DecisionTreeClassifier", "RandomForestClassifier", "BaggingClassifier"]


for i, v in enumerate(data_list):
    x, y = v
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state= 337, shuffle= True, stratify=y 
    )
    
    for j, v2 in enumerate(model_list):
        aaa = v2
        model = BaggingClassifier(aaa,
                          n_estimators= 20,
                          n_jobs= -1,
                          random_state= 337,
                          bootstrap= True)
        model.fit(x_train, y_train)
        print('==========', data_name[i], '==========')
        print('모델 이름 : ', model_name[j])
        
        y_pred = model.predict(x_test)
        print('model.score : ', model.score(x_test, y_test))
        print('acc : ', accuracy_score(y_test, y_pred))
        
        
# ========== 아이리스 ==========
# 모델 이름 :  DecisionTreeClassifier
# model.score :  0.9666666666666667
# acc :  0.9666666666666667
# ========== 아이리스 ==========
# 모델 이름 :  RandomForestClassifier
# model.score :  0.9666666666666667
# acc :  0.9666666666666667
# ========== 아이리스 ==========
# 모델 이름 :  BaggingClassifier
# model.score :  0.9666666666666667
# acc :  0.9666666666666667
# ========== 캔서 ==========
# 모델 이름 :  DecisionTreeClassifier
# model.score :  0.9298245614035088
# acc :  0.9298245614035088
# ========== 캔서 ==========
# 모델 이름 :  RandomForestClassifier
# model.score :  0.9385964912280702
# acc :  0.9385964912280702
# ========== 캔서 ==========
# 모델 이름 :  BaggingClassifier
# model.score :  0.9035087719298246
# acc :  0.9035087719298246
# ========== 디지트 ==========
# 모델 이름 :  DecisionTreeClassifier
# model.score :  0.9555555555555556
# acc :  0.9555555555555556
# ========== 디지트 ==========
# 모델 이름 :  RandomForestClassifier
# model.score :  0.9777777777777777
# acc :  0.9777777777777777
# ========== 디지트 ==========
# 모델 이름 :  BaggingClassifier
# model.score :  0.9722222222222222
# acc :  0.9722222222222222
# ========== 와인 ==========
# 모델 이름 :  DecisionTreeClassifier
# model.score :  1.0
# acc :  1.0
# ========== 와인 ==========
# 모델 이름 :  RandomForestClassifier
# model.score :  1.0
# acc :  1.0
# ========== 와인 ==========
# 모델 이름 :  BaggingClassifier
# model.score :  1.0
# acc :  1.0
# ========== 패치콥타입 ==========
# 모델 이름 :  DecisionTreeClassifier
# model.score :  0.9664724662874452
# acc :  0.9664724662874452


