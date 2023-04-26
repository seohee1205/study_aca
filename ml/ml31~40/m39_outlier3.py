import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 10000, 210, 420, 350]])
aaa = np.transpose(aaa)
# print(aaa.shape)  # (13, 2)

# [실습] outlier1을 이용해서 이상치를 찾아라
# 해결책 1: 컬럼을 for로 2번 돌리기
# 해결책 2: dataframe 통째로 함수로 받아들여서 returen하게 수정

def outliers(data_out):
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        outlier_indices = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))[0]
        whereoi = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))
        if outlier_indices.size > 0:
            print(i+1,"번째 컬런의 이상치 :", data_out[outlier_indices,i],'\n',' 이상치의 위치 :', whereoi[0])
        else:
            print(i+1,"번째 컬런 이상치 없음")

outliers(aaa)