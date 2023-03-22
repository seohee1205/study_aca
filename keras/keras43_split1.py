import numpy as np

dataset = np.array(range(1, 11))    # 1부터 10까지
timesteps = 5           # 5개씩 잘라라

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):           # 반복 횟수  10 - 5 + 1 = 6 
        subset = dataset[i : (i + timesteps)]               # 반복할 내용 0부터 0+5 즉 0부터 5까지의 데이터셋을 섭셋에 저장
        aaa.append(subset)                                  # 섭셋을 aaa에 추가해라
    return np.array(aaa)                    

# aaa라는 리스트 공간을 만들고
# for 반복할 거야, (6번을) , 

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)    # (6, 5)
