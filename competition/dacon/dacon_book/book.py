import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


#1. 데이터
path = 'd:/study_data/_data/dacon_book/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

# 결측치 확인
# print(train_csv.info())

x = train_csv.drop(['Book-Rating'], axis = 1)
# print(x)     # [871393 rows x 8 columns]

y = train_csv['Book-Rating']
# print(y)    # Name: Book-Rating, Length: 871393, dtype: int64

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 789
)

# print(x_train.shape, y_train.shape)  # (697114, 8) (697114,)
# print(x_test.shape, y_test.shape)   # (174279, 8) (174279,)

# Surprise 라이브러리용 Reader 및 Dataset 객체 생성
reader = Reader(rating_scale=(0, 10))
train = Dataset.load_from_df(train_csv[['User-ID', 'Book-ID', 'Book-Rating']], reader)
train = train.build_full_trainset()


# SVD 모델 훈련
model = SVD(n_factors=200,
            n_epochs = 1000,
            lr_all = 0.001,
            reg_all = 0.01)
model.fit(train)

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
testset = list(zip(x_test['User-ID'], x_test['Book-ID'], x_test['Book-Rating']))
predicted_ratings = model.test(testset)

# MSE 계산
mse = mean_squared_error([pred.r_ui for pred in predicted_ratings], [pred.est for pred in predicted_ratings])
print("MSE:", mse)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

# Prediction
submit = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

submit['Book-Rating'] = test_csv.apply(lambda row: model.predict(row['User-ID'], row['Book-ID']).est, axis=1)


#time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# Submission
save_path = 'd:/study_data/_save/dacon_book/'
submit.to_csv(save_path + date + '_sample_submission.csv', index=True)

