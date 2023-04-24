# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# m33_2 결과를 뛰어넘을 것

# parameters = {
#     {"_estimators": [100, 200, 300],
#      "learning_rate": [0.1, 0.3, 0.001, 0.01],
#     "max_depth": [4, 5, 6]},
#     {"_estimators": [90, 100, 110],
#     "learning_rate": [0.1, 0.001, 0.01],
#     "max _depth": [4,5,6],
#     "colsample_bytree": [0.6, 0.9, 1]},
#     {"_estimators": [90, 110],
#     "learning rate": [0.1, 0.001, 0.5],
#     "max _depth": [4,5,6],
#     "colsample _bytree": [0.6, 0.9, 1]},
#     {"colsample_bylevel": [0.6, 0.7, 0.9]}
# }
# n_jobs = -1
# tree_method ='gpu_hist'
# predictor ='gpu_predictor'
# gpu_id =0,



from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.datasets import mnist

# 데이터 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 및 PCA
x = np.concatenate((x_train, x_test), axis=0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
pca = PCA(n_components=0.95)
x = pca.fit_transform(x)

# 학습 데이터와 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y_train, test_size=0.2, random_state=123)

# xgboost 모델 학습
parameters = {
    "_estimators": [100, 200, 300],
    "learning_rate": [0.1, 0.3, 0.001, 0.01],
    "max_depth": [4, 5, 6],
    "colsample_bytree": [0.6, 0.9, 1],
    "colsample_bylevel": [0.6, 0.7, 0.9]
}
best_acc = 0.9648
model = xgb.XGBClassifier(
    n_jobs=-1,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0
)
for i in range(100):
    # 조기 종료 기능을 사용하여 학습
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        verbose=False,
        **parameters
    )
    # 모델 평가
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy after {model.best_iteration + 1} rounds: {acc:.4f}")
    # 최고 정확도 갱신 및 조기 종료
    if acc >= best_acc:
        print(f"Early stopping after {model.best_iteration + 1} rounds with accuracy {acc:.4f}")
        break




