import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action = 'ignore')
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
data_list = [load_iris(return_X_y= True), load_breast_cancer(return_X_y= True), 
             load_wine(return_X_y= True),]

model_list = [LinearSVC(), LogisticRegression(),
              DecisionTreeClassifier(), RandomForestClassifier(),]

data_name_list = ['아이리스 : ',
                  '브레스트 캔서 : ',
                  '와인 : ',]

model_name_list = ['LinearSVC : ',
                   'LogisticRegression : ',
                   'DecisionTreeClassifier :',
                   'RF : ',]

#2. 모델
for i, value in enumerate(data_list):
    x, y = value 
    # print(x.shape, y.shape)
    print("=====================")    
    print(data_name_list[i])
    
    for j, value2 in enumerate(model_list):
        model= value2
        #3. 컴파일, 훈련
        model.fit(x, y)
        #4. 평가, 예측
        results = model.score(x, y)      
        print(model_name_list[j], "model.score : ", results)
        y_predict = model.predict(x)
        acc = accuracy_score(y, y_predict)
        print(model_name_list[j], "accuracy_score : ", acc)


# =====================
# 아이리스 : 
# LinearSVC :  model.score :  0.9666666666666667   
# LinearSVC :  accuracy_score :  0.9666666666666667
# LogisticRegression :  model.score :  0.9733333333333334   
# LogisticRegression :  accuracy_score :  0.9733333333333334
# DecisionTreeClassifier : model.score :  1.0
# DecisionTreeClassifier : accuracy_score :  1.0
# RF :  model.score :  1.0   
# RF :  accuracy_score :  1.0
# =====================
# 브레스트 캔서 :
# LinearSVC :  model.score :  0.9244288224956063
# LinearSVC :  accuracy_score :  0.9244288224956063
# LogisticRegression :  model.score :  0.9472759226713533
# LogisticRegression :  accuracy_score :  0.9472759226713533
# DecisionTreeClassifier : model.score :  1.0
# DecisionTreeClassifier : accuracy_score :  1.0
# RF :  model.score :  1.0
# RF :  accuracy_score :  1.0
# =====================
# 와인 :
# LinearSVC :  model.score :  0.9325842696629213
# LinearSVC :  accuracy_score :  0.9325842696629213
# LogisticRegression :  model.score :  0.9662921348314607
# LogisticRegression :  accuracy_score :  0.9662921348314607
# DecisionTreeClassifier : model.score :  1.0
# DecisionTreeClassifier : accuracy_score :  1.0
# RF :  model.score :  1.0
# RF :  accuracy_score :  1.0