import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

x, y = make_blobs(
    n_samples= 50,
    centers= 2,         # 중심 클러스터 개수 (= y의 라벨)
    cluster_std=1,     # 클러스터 표준편차
    random_state= 337)
# 가우시안 정규분포 샘플 생성

print(x)
print(y)
print(x.shape, y.shape)     # (50, 2) (50,)

fig, ax = plt.subplots(2, 2, figsize = (12, 8))

ax[0, 0].scatter(x[:, 0], x[:, 1],
            c=y,
            edgecolors= 'black',
            )
ax[0, 0].set_title("오리지널")

scaler = QuantileTransformer(n_quantiles= 50)
x_trans = scaler.fit_transform(x)
ax[0, 1].scatter(x_trans[:, 0], x_trans[:, 1],
            c=y,
            edgecolors= 'black',
            )
ax[0, 1].set_title(type(scaler).__name__)

scaler = PowerTransformer()
x_trans = scaler.fit_transform(x)
ax[1, 0].scatter(x_trans[:, 0], x_trans[:, 1],
            c=y,
            edgecolors= 'black',
            )
ax[1, 0].set_title(type(scaler).__name__)


scaler = StandardScaler()
x_trans = scaler.fit_transform(x)
ax[1, 1].scatter(x_trans[:, 0], x_trans[:, 1],
            c=y,
            edgecolors= 'black',
            )
ax[1, 1].set_title(type(scaler).__name__)

plt.show()

