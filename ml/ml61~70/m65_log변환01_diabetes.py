from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1. 데이터셋
datasets = load_diabetes()
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

df['target'].hist(bins=50)  # bins=50은 분위수를 50개씩 잘랐다는 소리
plt.show()

y = df['target']
x = df.drop(['target'], axis= 1)

# # ##################### x population 로그변환 #####################
x['bmi'] = np.log1p(x['bmi'])      # 지수변환  np.explm
x['s3'] = np.log1p(x['s3'])      # 지수변환  np.explm

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size= 0.8, random_state= 337
)

# ##################### y 로그변환 #####################
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
####################################################

#2. 모델
model = RandomForestRegressor(random_state=337)

#3. 컴파일, 훈련
model.fit(x_train, y_train_log)

#4 평가, 예측
score = model.score(x_test, y_test_log)
r2 = r2_score(y_test, np.expm1(model.predict(x_test))) #로그변환된 값을 다시 지수변환 하고 비교해야한다

print('score : ', score)
print('r2 :', r2)

print("로그 -> 지수 r2 : ", r2_score(y_test, np.expm1(model.predict(x_test))))

# score :  0.3483379381533718
# r2 : 0.3829856840673078
# 로그 -> 지수 r2 :  0.3829856840673078
