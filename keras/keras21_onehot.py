# [과제]
# 3가지 원핫인코딩 방식을 비교할 것
#1. pandas의 get_dummies
#2. keras의 to_categorical
#3. sklearn의 OneHotEncoder
# 미세한 차이를 정리하시오


# 1. pandas의 get_dummies
# 기능: 범주형 데이터를 one-hot encoding으로 변환
# 장점: 간단하고 쉽게 사용할 수 있으며, 범주형 데이터에 대한 처리를 빠르게 수행 가능
# 단점: 범주형 변수의 수가 많아질 경우 처리 속도가 느려질 수 있음, 또한 get_dummies 는 test 데이터의 특성을 학습하지 않는다.

import pandas as pd
y=pd.get_dummies(y)
print(y.shape)


# 2. keras의 to_categorical
# 기능: 정수형 레이블을 one-hot encoding으로 변환
# 장점: 간단하고 쉽게 사용할 수 있으며, 다중 분류 문제에서 자주 사용됨
# 단점: 범주형 데이터를 변환할 때는 사용할 수 없으며, keras 라이브러리를 사용해야 함. 
# 0 데이터가 없을 경우 0의 열을 생성시켜 자동으로 0 데이터를 다 넣어버림

from tensorflow.keras.utils import to_categorical #tensorflow 빼도 가능.
y = to_categorical(y)
print(y.shape)


# 3. sklearn의 OneHotEncoder
# 기능: 범주형 데이터를 one-hot encoding으로 변환
# 장점: 대용량 데이터에 대한 처리가 가능하며, pandas의 get_dummies보다 더 많은 옵션과 기능을 제공함
# 단점: fit 메소드를 호출하여 범주형 변수의 유니크한 값들을 먼저 파악해야 하며, 이에 따른 메모리 사용량이 증가함

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y.shape)


# 결론적으로, pandas의 get_dummies는 간단하고 빠르게 범주형 데이터를 처리할 수 있는 장점이 있으며, 
# keras의 to_categorical은 다중 분류 문제에서 유용합니다. 
# 하지만 sklearn의 OneHotEncoder는 대용량 데이터에 대한 처리와 더 다양한 옵션을 제공하므로 유연한 처리가 가능합니다.

