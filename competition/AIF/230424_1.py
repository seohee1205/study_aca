import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error
# import datetime

# path = 'd:/study_data/_data/aif/초미세먼지/'
# save_path = 'd:/study_data/_save/aif/초미세먼지/'

# train_path = 'd:/study_data/_data/aif/초미세먼지/TRAIN/'
# train_AWS_path = 'd:/study_data/_data/aif/초미세먼지/TRAIN_AWS/'
# test_INPUT_path = 'd:/study_data/_data/aif/초미세먼지/TEST_INPUT/'
# test_AWS_path = 'd:/study_data/_data/aif/초미세먼지/TEST_AWS/'
# meta_path = 'd:/study_data/_data/aif/초미세먼지/META/'

import pandas as pd
import os

path='d:/study_data/_data/aif/초미세먼지/'
path_list=os.listdir(path)
print(f'datafolder_list:{path_list}')

meta='/'.join([path,path_list[1]])
meta_list=os.listdir(meta)
test_aws='/'.join([path,path_list[2]])
test_aws_list=os.listdir(test_aws)
test_input='/'.join([path,path_list[3]])
test_input_list=os.listdir(test_input)
train='/'.join([path,path_list[4]])
train_list=os.listdir(train)
train_aws='/'.join([path,path_list[5]])
train_aws_list=os.listdir(train_aws)

print(f'META_list:{meta_list}')
awsmap=pd.read_csv('/'.join([meta,meta_list[0]]))
awsmap=awsmap.drop(awsmap.columns[-1],axis=1)
pmmap=pd.read_csv('/'.join([meta,meta_list[1]]))
pmmap=pmmap.drop(pmmap.columns[-1],axis=1)
print(awsmap)
print(pmmap)



import pandas as pd
import os

# 데이터 파일이 저장된 디렉토리 경로
data_path = "/path/to/data/files/"

# 해당 디렉토리 내의 파일 목록을 가져옴
file_list = glob.glob(os.path.join(data_path, "*.csv"))

# 각 파일의 데이터를 순차적으로 불러와서 하나의 데이터 프레임으로 반환
df = pd.concat([pd.read_csv(f) for f in file_list])