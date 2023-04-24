# [실습]
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성후
# 모델을 돌려서 결과 도출
# 기초모델들과 성능비교

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb

#1. 데이터

data_list = [load_iris(), load_breast_cancer(), load_digits(), load_wine()]
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier(objective='multi:softmax')]

for i in range(len(data_list)):
    data = data_list[i]
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state= 337, stratify=y
    )

    for j, model in enumerate(model_list): 
        # 모델 훈련
        model.fit(x_train, y_train)

        # feature importance 계산
        feature_importance = model.feature_importances_
        feature_importance_percent = feature_importance / feature_importance.sum()

        # 중요도가 낮은 컬럼 제거
        threshold = np.percentile(feature_importance_percent, 25) # 하위 25%
        feature_idx = np.where(feature_importance_percent >= threshold)[0]
        selected_x_train = x_train[:, feature_idx]
        selected_x_test = x_test[:, feature_idx]

        # XGBClassifier 모델에서 num_class 파라미터 추가 
        if isinstance(model, XGBClassifier):
            model.fit(selected_x_train, y_train, n_classes=len(np.unique(y)))
        else:
            model.fit(selected_x_train, y_train)
            
        train_acc = model.score(selected_x_train, y_train)
        test_acc = model.score(selected_x_test, y_test)

        # 결과 출력
        print(f"Dataset: {data_list[i].DESCR.splitlines()[0]}")
        print(f"Model: {type(model).__name__}")
        print(f"Selected Features: {len(feature_idx)} / {x_train.shape[1]}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("=" * 50)




































# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier

# #1. 데이터

# data_list = [load_iris(), load_breast_cancer(), load_digits(), load_wine()]
# model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]
# scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()] 

# for i in range(len(data_list)):
#     x, y = data_list[i](return_X_y=True)
#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, train_size= 0.8, random_state= 337, stratify=y
#     )
    
#     for j, value2 in enumerate(scaler_list):
#         for k, model in enumerate(model_list):
#             # 모델 훈련
#             model.fit(x_train, y_train)
            
#             model = 
            


#2. 모델구성


#결과 비교
#예)
#1. DecisionTree
# 기존 acc: 0.000
# 컬럼삭제후 acc: 

#2. RandomForest
# 기존 acc: 0.000
# 컬럼삭제후 acc: 

#3. GradientDecentBoosting
#4. XGBoost

