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
print(train_csv.info())

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
model.fit(x_train, y_train)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

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
submit.to_csv(save_path + date + '_sample_submission.csv', index=False)


###################
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
font_path = "C:/Windows/Fonts/malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

train_df = pd.read_csv('train.csv')
train_df

# 결측치 확인
train_df.isnull().sum()

ds = train_df['Book-Rating'].value_counts().reset_index()

ds.columns = ['value', 'count']

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='value', y='count', data=ds, ax=ax)
ax.set_title('Ranking 분포')
ax.set_xlabel('Ranking')
ax.set_ylabel('Count')
plt.show()

ds = train_df['Year-Of-Publication'].value_counts().reset_index()
ds.columns = ['value', 'count']
ds['value'] = ds['value'].astype(str) + ' year'
ds = ds.sort_values('count', ascending=False).head(50)

plt.figure(figsize=(8, 9))
sns.barplot(data=ds, x='count', y='value', orient='h')
plt.title('Top 50 Year-Of-Publication')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

ds = train_df['Book-Author'].value_counts().reset_index()
ds.columns = ['author', 'count']
ds = ds.sort_values('count', ascending=False).head(50)

plt.figure(figsize=(8, 9))
sns.barplot(x='count', y='author', data=ds, orient='h')
plt.title('Top 50 Book-Author')
plt.show()

ds = train_df['Book-Title'].value_counts().reset_index()
ds.columns = ['Book-Title', 'count']
ds = ds.sort_values('count', ascending=False).head(50)

plt.figure(figsize=(8, 9))
sns.barplot(data=ds, x='count', y='Book-Title')
plt.title('Top 50 Book-Title')
plt.xlabel('count')
plt.ylabel('Book-Title')
plt.show()

plt.figure(figsize=(7, 6))
sns.histplot(train_df['Age'], bins=100)
plt.title('Age 분포')
plt.xlabel('나이')
plt.ylabel('빈도수')
plt.show()

data = train_df.groupby('Book-Rating')['Age'].mean().reset_index()

plt.figure(figsize=(8,7))
sns.barplot(x='Book-Rating', y='Age', data=data)
plt.title('Book-Rating 별 평균 나이')
plt.show()

users = train_df['User-ID'].value_counts().reset_index()
users.columns = ['User-ID', 'evaluation_count']
users = users.sort_values('evaluation_count', ascending=False)

plt.figure(figsize=(8, 9))
sns.barplot(x='evaluation_count', y='User-ID', data=users.head(50))
plt.title('Top 50 book reviewers')
plt.xlabel('Evaluation Count')
plt.ylabel('User-ID')
plt.show()

books = train_df['Book-Title'].value_counts().reset_index()
books.columns = ['Book-Title', 'book_evaluation_count']
df = pd.merge(train_df, books)
mean_df = df[df['book_evaluation_count']>100]
mean_df = mean_df.groupby('Book-Title')['Book-Rating'].mean().reset_index().sort_values('Book-Rating', ascending=False)

plt.figure(figsize=(10, 9))
sns.barplot(data=mean_df.tail(50), x='Book-Rating', y='Book-Title')
plt.title('평균 평점이 가장 높은 상위 50개 도서')
plt.xlabel('평균 평점')
plt.ylabel('도서 제목')
plt.show()

books = df['Publisher'].value_counts().reset_index()
books.columns = ['Publisher', 'Publisher_evaluation_count']
df = pd.merge(df, books)
mean_df = df[df['Publisher_evaluation_count']>100]
mean_df = mean_df.groupby('Publisher')['Book-Rating'].mean().reset_index().sort_values('Book-Rating', ascending=False)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 9))
sns.barplot(x='Book-Rating', y='Publisher', data=mean_df.head(50), orient='h')
plt.title('Top 50 Publishers with highest avarage Book-Rating', fontsize=16)
plt.show()


books = df['Book-Author'].value_counts().reset_index()
books.columns = ['Book-Author', 'author_evaluation_count']
df = pd.merge(df, books)

mean_df = df[df['author_evaluation_count']>100]
mean_df = mean_df.groupby('Book-Author')['Book-Rating'].mean().reset_index().sort_values('Book-Rating', ascending=False)

top_50_mean_df = mean_df.head(50)

plt.figure(figsize=(10,9))
sns.barplot(x='Book-Rating', y='Book-Author', data=top_50_mean_df, orient='h')
plt.title('Top 50 Book-Author with highest avarage Book-Rating')
plt.show()

'''
