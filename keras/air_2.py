import pandas as pd
from sklearn.ensemble import IsolationForest
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

# 훈련 데이터를 로드합니다.
path = 'd:/study_data/_data/air/dataset/'
save_path = 'd:/study_data/_save/air/'
train_data = pd.read_csv(path + 'train_data.csv', index_col = 0)

# Isolation Forest 모델을 훈련합니다.
model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.1))
model.fit(train_data)

# 테스트 데이터를 로드합니다.
test_data = pd.read_csv(path + 'test_data.csv', index_col = 0)

# 테스트 데이터에서 이상값을 예측합니다.
predictions = model.predict(test_data)

# 이상값이 아닌 데이터를 필터링합니다.
anomalies = test_data[predictions == -1]

# 이상값을 출력합니다.
print(anomalies)

submission = pd.read_csv(path + 'answer_sample.csv')
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})
submission.to_csv(save_path + date + 'submission.csv', index=False)