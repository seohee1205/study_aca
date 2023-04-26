import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 700,
                8, 9, 10, 11, 12, 50])
aaa = aaa.reshape(-1, 1)        # 원하는 형태가 2차원이기 때문에

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination= .1)  # 전체 데이터 중 몇 %를 이상치로 할 건지

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)

