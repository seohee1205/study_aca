import numpy as np
# aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
aaa = np.array([2, 3, 4, 5, 6, 7, 8, 1000, -10, 9, 10, 11, 12, 50])   # -10 바꿔놓은 거
# 소트 안 해도 됨(np.where에서 알아서 위치 찾음)

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
