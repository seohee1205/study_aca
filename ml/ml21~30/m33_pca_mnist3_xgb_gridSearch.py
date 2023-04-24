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



