# PCA = 차원 축소의 개념. (컬런 압축의 개념)
# 일반적으로 x만 PCA를 적용한다.
# x만 사용하기 때문에 비지도 학습으로 분류된다. (차원축소한 결과를 y로 볼 수 있기 때문)
# 스케일링(전처리) 개념으로 볼 수도 있다.

# [실습]
# for문 써서 한번에 돌려
# 기본결과 : 0.23131244
# 차원 1개 축소: 0.3341432
# 차원 2개 축소: 0.423414
# ...

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_digits()
print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
# ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 
# 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 
# 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6',
# 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 
# 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 
# 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 
# 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 
# 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7',
# 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 
# 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 
# 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 
# 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 
# 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)    # (1797, 64) (1797,)

for i in range(64, 0, -1):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestRegressor(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")


# n_coponets=64,  결과: 0.7812653547686522 
# n_coponets=63,  결과: 0.7808869348730036 
# n_coponets=62,  결과: 0.7825229052449436 
# n_coponets=61,  결과: 0.7830082898535371 
# n_coponets=60,  결과: 0.7804417723865773 
# n_coponets=59,  결과: 0.7815468743102824 
# n_coponets=58,  결과: 0.7834785249534303 
# n_coponets=57,  결과: 0.7799246055918991 
# n_coponets=56,  결과: 0.7850743791438233 
# n_coponets=55,  결과: 0.7866179339813368 
# n_coponets=54,  결과: 0.7830929364091427 
# n_coponets=53,  결과: 0.7819365239116809 
# n_coponets=52,  결과: 0.7825230818126441 
# n_coponets=51,  결과: 0.7843362908423163 
# n_coponets=50,  결과: 0.7826212181405655 
# n_coponets=49,  결과: 0.7869966716988461 
# n_coponets=48,  결과: 0.7854424874857642 
# n_coponets=47,  결과: 0.7863338012377397 
# n_coponets=46,  결과: 0.7834758764379233 
# n_coponets=45,  결과: 0.780510704416841 
# n_coponets=44,  결과: 0.7856459994173266 
# n_coponets=43,  결과: 0.7882129406467675 
# n_coponets=42,  결과: 0.7876267358812052 
# n_coponets=41,  결과: 0.7826213947082661 
# n_coponets=40,  결과: 0.7834148546406405 
# n_coponets=39,  결과: 0.785143734936568 
# n_coponets=38,  결과: 0.7887464222969692 
# n_coponets=37,  결과: 0.7859559816722728 
# n_coponets=36,  결과: 0.7916499721905872 
# n_coponets=35,  결과: 0.7983378269813104 
# n_coponets=34,  결과: 0.7994447652091003 
# n_coponets=33,  결과: 0.7991570657979536 
# n_coponets=32,  결과: 0.7993036169893442 
# n_coponets=31,  결과: 0.7984811293270122 
# n_coponets=30,  결과: 0.7985104748788304 
# n_coponets=29,  결과: 0.802688207926124 
# n_coponets=28,  결과: 0.8043194462836913 
# n_coponets=27,  결과: 0.8053746501752435 
# n_coponets=26,  결과: 0.8069501284530021 
# n_coponets=25,  결과: 0.8088549408056784 
# n_coponets=24,  결과: 0.8083248492553258 
# n_coponets=23,  결과: 0.809925329519471 
# n_coponets=22,  결과: 0.8075784975854368 
# n_coponets=21,  결과: 0.8124355572035208 
# n_coponets=20,  결과: 0.8129539246585622 
# n_coponets=19,  결과: 0.8108251185210689 
# n_coponets=18,  결과: 0.8146486920747588 
# n_coponets=17,  결과: 0.8229092353735731 
# n_coponets=16,  결과: 0.8202788357125831 
# n_coponets=15,  결과: 0.817956829197235 
# n_coponets=14,  결과: 0.8195279992231022 
# n_coponets=13,  결과: 0.8225419392430543 
# n_coponets=12,  결과: 0.8237070035578391 
# n_coponets=11,  결과: 0.8203734053729551 
# n_coponets=10,  결과: 0.8184514306397931 
# n_coponets=9,  결과: 0.8140142136998879 
# n_coponets=8,  결과: 0.791123447307784 
# n_coponets=7,  결과: 0.8104294656178546 
# n_coponets=6,  결과: 0.7492982316744798 
# n_coponets=5,  결과: 0.7421507358458916 
# n_coponets=4,  결과: 0.6724708707436149 
# n_coponets=3,  결과: 0.6199968570949317 
# n_coponets=2,  결과: 0.39008914903196756 
# n_coponets=1,  결과: -0.48619311209400484 
