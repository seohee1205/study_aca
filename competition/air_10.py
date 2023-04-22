import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# 훈련 데이터 및 테스트 데이터 로드
path='d:/study_data/_data/air/dataset/'
save_path= 'd:/study_data/_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 데이터 전처리
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])
# print(train_data.columns)

# lof모델 적용 피처
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# 학습 데이터를 훈련 세트와 검증 세트로 나누기
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 5050)

# 데이터 정규화
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# lof사용하여 이상치 탐지
n_neighbors = 37
contamination = 0.045
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, leaf_size=20)
y_pred_train_tuned = lof.fit_predict(X_train)

# 이상치 탐지
test_data_lof = scaler.transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]


# 모델







submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())
#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)