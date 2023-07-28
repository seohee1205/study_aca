import autokeras as ak
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Iris 데이터셋 로드
data = load_iris()
x = data.data
y = data.target

# 훈련 세트와 테스트 세트로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# AutoKeras 분류기 모델 생성
clf = ak.StructuredDataClassifier(max_trials=1, overwrite=False)  # 이미지 분류기 모델

# 모델 훈련
clf.fit(x_train, y_train, epochs=10)

# 모델 평가
results = clf.evaluate(x_test, y_test)
print('결과:', results)

# 결과: [0.5355831980705261, 0.800000011920929]