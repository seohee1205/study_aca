import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

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

# lof모델 적용 피처
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# 학습 데이터를 훈련 세트와 검증 세트로 나누기
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 5050)

# 데이터 정규화
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
test_data_norm = scaler.transform(test_data[features])

# lof사용하여 이상치 탐지
n_neighbors = 37
contamination = 0.048
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, leaf_size=20)
y_pred_train_lof = lof.fit_predict(X_train_norm)
y_pred_test_lof = lof.fit_predict(test_data_norm)

# 이상치 탐지
y_pred_test_lof = lof.fit_predict(test_data_norm)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]


# 이상치 제거
train_outlier_indices = np.where(y_pred_train_lof == -1)
X_train_norm = np.delete(X_train_norm, train_outlier_indices, axis=0)
X_train = X_train.drop(X_train.index[train_outlier_indices], axis=0)

val_outlier_indices = np.where(y_pred_test_lof == -1)
X_val_norm = np.delete(X_val_norm, val_outlier_indices, axis=0)
X_val = X_val.drop(X_val.index[val_outlier_indices], axis=0)


# 모델
model = Sequential()
model.add(Dense(512, activation='swish', input_dim=X_train_norm.shape[1]))
model.add(Dense(300, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(60, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(X_train_norm.shape[1], activation='linear'))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=55)

history = model.fit(X_train_norm, X_train_norm, epochs=2000, 
                    batch_size=22, 
                    validation_data=(X_val_norm, X_val_norm), callbacks=[es])

model.save("d:/study_data/_save/air/air0421_0325.h5")


# 평가
test_preds = model.predict(test_data_norm)
errors = np.mean(np.power(test_data_norm - test_preds, 2), axis=1)
y_pred = np.where(errors >= np.percentile(errors, 96), 1, 0)

submission['label'] = y_pred
# submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())
print(submission['label'].value_counts())

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)