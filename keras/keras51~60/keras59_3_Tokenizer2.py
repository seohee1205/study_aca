from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'

token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index) 
# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, 
#  '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '배환희다': 11, '멋있다': 12, '얘기해부아': 13}
# 정렬기준 : 최빈값 -> 앞에서부터

print(token.word_counts)
# OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을
# ', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('지구용사', 1), ('배환희
# 다', 1), ('멋있다', 1), ('또', 2), ('얘기해부아', 1)])

x = token.texts_to_sequences([text1, text2])    
print(x)                    # [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]
print(type(x))              # <class 'list'>



######### 1. to_categorical ##########
x = x[0] + x[1]
print(x)

# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape)          # (18, 14)

######### 2. get_dummies ##########
# x = pd.get_dummies(np.array(x).reshape(11,))
# x = pd.get_dummies(np.array(x).ravel())
# print(x)
# print(x.shape)      # (18, 13)

# x = pd.get_dummies(x[0])
# print(x)


######### 3. 사이킷런 onehot ##########
# 2차원으로 받아들여야 한다.
from sklearn.preprocessing import OneHotEncoder
import numpy as np
ohe = OneHotEncoder()
x = ohe.fit_transform(np.array(x).reshape(-1, 1)).toarray()
print(x)
print(x.shape)          # (18, 13)