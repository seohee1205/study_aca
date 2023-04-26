import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

# 이상치를 찾아주는 함수
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    
    print("1사분위 : ", quartile_1)                  # 4
    print("q2 : ", q2)                              # 7 
    print("3사분위 : ", quartile_3)                  # 10
    iqr = quartile_3 - quartile_1           
    print("iqr : ", iqr)                            # 10 - 4 = 6
    lower_bound = quartile_1 - (iqr * 1.5)          # -5 = 4 - (6 * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)          # 19 = 10 + (6 * 1.5)
    return np.where((data_out > upper_bound) |      # | = or , (이거이면 이 값을 넘겨라)
                    (data_out < lower_bound))       # 19보다 크거나, -5보다 작으면 반환

outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)


import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()


# 이 코드는 주어진 데이터에서 이상치(outlier)를 찾는 함수를 구현한 것
# 이상치란 일반적인 데이터 분포에서 벗어나서 튀어나온 값으로, 
# 다른 데이터와 매우 차이가 나는 값

# 이 함수는 먼저 데이터의 사분위수(quartile)를 계산함 
# 사분위수란 데이터를 크기순으로 정렬했을 때 1/4, 2/4(중앙값), 3/4 위치에 있는 값을 의미
# 이때 1사분위는 전체 데이터의 25% 지점에 있는 값, 
# 3사분위는 전체 데이터의 75% 지점에 있는 값임

# 사분위수를 계산한 뒤, 이를 이용해 IQR(interquartile range)를 계산함
# IQR은 3사분위와 1사분위의 차이로, 데이터의 중간 50% 범위를 나타내는 값임

# 그 다음, 이상치의 경계를 계산함
# 이상치는 일반적으로 IQR의 1.5배 이상 벗어나는 값을 의미함
# 따라서 상한선(upper bound)은 3사분위 값에 IQR의 1.5배를 더한 값이고, 
# 하한선(lower bound)은 1사분위 값에서 IQR의 1.5배를 뺀 값임

# 마지막으로, numpy의 where 함수를 사용하여 이상치에 해당하는 위치를 찾아서 반환함
# 이때 반환되는 값은 이상치의 인덱스를 담은 numpy 배열임

# 실제로 이 코드를 실행하면 [-10, 50] 이상치에 해당하는 위치 [ 0 12]가 출력됨