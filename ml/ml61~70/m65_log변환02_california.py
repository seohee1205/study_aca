from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터셋
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns= datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

# df.info()
# print(df.describe())

# df['Population'].boxplot()    # 이거 안 돼
# df['Population'].plot.box()   # 이거 써
# plt.show()

df.hist(bins=50)    # 히스토그램 그리기
plt.show()          

df['Population'].hist(bins = 50)
plt.show()

df['target'].hist(bins=50)    # bins=50은 분위수를 50개씩 잘랐다는 소리
plt.show()

y = df['target']
x = df.drop(['target'], axis= 1)

################## x Population 로그변환 ###################

 # 지수변환 np.exp1m

################## y 로그변환 ###################

y = np.log1p(y)

############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    shuffle = True,
                                                    random_state = 1234)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#2 모델
model = RandomForestRegressor(random_state = 1234)

#3 훈련
model.fit(x_train, y_train_log)

#4 평가, 예측
score = model.score(x_test, y_test_log)

print('score : ', score)

print("로그 -> 지수 r2 : ", r2_score(y_test, np.expm1(model.predict(x_test))))


# 로그 변환전
# score :  0.7977884836830872

# x 로그 변환후
# score :  0.7978889489982891

# y 로그 변환후
# score :  0.8216824110874738

# x, y 로그 변환후
# score :  0.8217758875704867

# 로그 -> 지수 r2 :  0.7952540050640855