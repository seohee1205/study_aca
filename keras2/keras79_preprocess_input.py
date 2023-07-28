from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions   # decode: 복호화
import numpy as np

model = ResNet50(weights = 'imagenet')
# model = ResNet50(weights = None)
# model = ResNet50(weights = '경로')

# path = 'D:\study_data\_data\cat_dog\PetImages\Dog\\6.jpg'
# path = 'D:\study_data\_data\수트.jpg'
## 본인들 사진을 넣어서 확인
path = 'D:\study_data\_data\나.jpg'


img = image.load_img(path, target_size= (224, 224))
print(img)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x1F8B55382B0>

x = image.img_to_array(img)
print("==================== image.img_to_array(img) ====================")
print(x, '\n', x.shape)     # (224, 224, 3)
print(np.min(x), np.max(x)) # 0.0 255.0

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
# print(x.shape)      # (1, 224, 224, 3)

x = np.expand_dims(x, axis = 0)
print(x.shape)      #  (1, 224, 224, 3)

#################### -155에서 155 사이로 정규화 ###################
print("==================== preprocess_input(x) ====================")

x = preprocess_input(x)
print(x.shape)      #  (1, 224, 224, 3)
print(np.min(x), np.max(x))      # -123.68 151.061

print("==================== model.predict(x) ====================")
x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)       #  (1, 1000)

print("결과는 : ", decode_predictions(x_pred, top=5)[0])


