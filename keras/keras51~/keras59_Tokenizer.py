from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

# 텍스트를 인식하기 위해 수치화함
token = Tokenizer()
token.fit_on_texts([text])

# print(token.word_index) # {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
#                         # 가장 많은 단어인 '마구'가 1, 그 다음 '매우'가 2, ... 
# print(token.word_counts)    # OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])
#                         # index 순
                        
x = token.texts_to_sequences([text])
# print(x)    # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]       -> 1행 11열
print(type(x))  # <class 'list'>

########### 원핫인코딩
##### 1. to_categorical #####
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape)      # (1, 11, 9)

##### 2. get_dummies #####
import pandas as pd
x = pd.get_dummies(x)
# print(x)
# 리스트를 넘파이로 바꾸기
import numpy as np
my_series = pd.Series(x)
dummies = pd.get_dummies(my_series, prefix='x')
numpy_array = dummies.values

print(numpy_array)

# 왜 리스트는 안 되는지
# 