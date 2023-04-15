# 분류 만들기

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)       # 1.0.2


#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 123, test_size= 0.2
)

scaler= RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
# model = RandomForestRegressor(n_jobs=4)
# allAlgorithms= all_estimators(type_filter= 'regressor')
allAlgorithms= all_estimators(type_filter= 'classifier')  # 분류일 때

print('allAlgorithms : ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms))      # 55

max_r2 = 0
max_name = '바보'

for (name, algorithm) in allAlgorithms:
    try:
        model= algorithm()
        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, "의 정답률  : ", results)

        if max_r2 < results:
            max_r2 = results
            max_name = name
        # y_predict = model.predict(x_test)
        # # print(y_test.dtype)     # float64
        # # print(y_predict.dtype)  # float64
        # aaa= r2_score(y_test, y_predict)
        # print("r2_Score : ", aaa)
    except:
        # print("반장 바보")
        print(name, 'error')
print("=============================")
print('최고모델 : ', max_name, max_r2)
print("=============================")






