import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[-10,2,3,4,5,6,700,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

outliers = EllipticEnvelope(contamination=.1)

for i, column in enumerate(aaa): 
    outliers.fit(column.reshape(-1, 1))  
    results = outliers.predict(column.reshape(-1, 1))
    outliers_save = np.where(results == -1)
    # print(outliers_save)
    # print(outliers_save[0])
    outliers_values = column[outliers_save] 
    
    print(f"{i+1}번째 컬런의 이상치 : {', '.join(map(str, outliers_values))}\n 이상치의 위치 : {', '.join(map(str, outliers_save))}")
    

# 1번째 컬런의 이상치 : 700, 50
#  이상치의 위치 : [ 6 12]
# 2번째 컬런의 이상치 : -70000, 1000
#  이상치의 위치 : [6 9]