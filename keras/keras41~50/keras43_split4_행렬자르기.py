import numpy as np

datasets= np.array(range(1, 41)).reshape(10, 4)
print(datasets)
print(datasets.shape)   # (10, 4)

# x_data = datasets[:, 3]
x_data = datasets[:, :-1]
y_data = datasets[:, -1]
print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)   # (10. 3) (10,)

timesteps = 3

############ x 만들기 ############
def split_x(dataset, timesteps):
    aaa = []    # aaa라는 빈 리스트를 만들어라
    for i in range(len(dataset) - timesteps ):        
        subset = dataset[i : (i + timesteps)]               
        aaa.append(subset)                                  
    return np.array(aaa)                    

 

x_data = split_x(x_data, timesteps)
print(x_data)
print(x_data.shape)    # (5, 5, 3)

################### y 만들기 #################
y_data = y_data[timesteps:]
print(y_data)
