# 그래프 그린다
# 1. value_counts => 쓰지마
# 2. np.arraund의 treutrn_counts 쓰지마
############## 3 grouphy 써, count() -> 써 #######################

# 4. plt.bar 로 그린다(quality 컬럼)

# 힌트
# 데이터개수(y축) = 데이터개수.주저리 주저리 ...    grouphy 


#[실습]Dancon_wine : ML활용 acc올리기
#결측치/ 원핫인코딩, 데이터분리, 스케일링/ 함수형,dropout
#다중분류 - softmax, categorical






count_data = train_csv.groupby('quality')['quality'].count()
print(count_data.index, count_data)

import matplotlib.pyplot as plt
plt.bar(count_data.index, count_data)
plt.show()