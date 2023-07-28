from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

# model_list = [VGG16, VGG19, ......]

# model = VGG16()
# model = VGG19()
# # ...

# model.trainable = False
# # model.summary()

# print("=========================")
# print("모델명 : ", "무슨 모델")
# print("전체 가중치 개수 : ", len(model.weights))
# print("훈련 가능 개수 : ", len(model.trainable.weights))

#################### 결과 출력 ####################
#### for문 쓰면 편하겠지

model_list = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
              DenseNet201, DenseNet121, DenseNet169, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2,
              MobileNetV3Small, MobileNetV3Large, NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB1,
              EfficientNetB7, Xception]

for model in model_list:
    model_name = model.__name__
    model_instance = model()
    model_instance.trainable = False
    
    print("=========================")
    print("모델명: ", model_name)
    print("전체 가중치 개수: ", len(model_instance.weights))
    print("훈련 가능 개수: ", len(model_instance.trainable_weights))


# =========================
# 모델명:  VGG16
# 전체 가중치 개수:  32
# 훈련 가능 개수:  0
# =========================
# 모델명:  VGG19
# 전체 가중치 개수:  38
# 훈련 가능 개수:  0
# =========================
# 모델명:  ResNet50
# 전체 가중치 개수:  320
# 훈련 가능 개수:  0
# =========================
# 모델명:  ResNet50V2
# 전체 가중치 개수:  272
# 훈련 가능 개수:  0
# =========================
# 모델명:  ResNet101
# 전체 가중치 개수:  626
# 훈련 가능 개수:  0
# =========================
# 모델명:  ResNet101V2
# 전체 가중치 개수:  544
# 훈련 가능 개수:  0
# =========================
# 모델명:  ResNet152
# 전체 가중치 개수:  932
# 훈련 가능 개수:  0
# =========================
# 모델명:  ResNet152V2
# 전체 가중치 개수:  816
# 훈련 가능 개수:  0
# =========================
# 모델명:  DenseNet201
# 전체 가중치 개수:  1006
# 훈련 가능 개수:  0
# =========================
# 모델명:  DenseNet121
# 전체 가중치 개수:  606
# 훈련 가능 개수:  0
# =========================
# 모델명:  DenseNet169
# 전체 가중치 개수:  846
# 훈련 가능 개수:  0
# =========================
# 모델명:  InceptionV3
# 전체 가중치 개수:  378
# 훈련 가능 개수:  0
# =========================
# 모델명:  InceptionResNetV2
# 전체 가중치 개수:  898
# 훈련 가능 개수:  0
# =========================
# 모델명:  MobileNet
# 전체 가중치 개수:  137
# 훈련 가능 개수:  0
# =========================
# 모델명:  MobileNetV2
# 전체 가중치 개수:  262
# 훈련 가능 개수:  0
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# =========================
# 모델명:  MobileNetV3Small
# 전체 가중치 개수:  210
# 훈련 가능 개수:  0
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# =========================
# 모델명:  MobileNetV3Large
# 전체 가중치 개수:  266
# 훈련 가능 개수:  0
# =========================
# 모델명:  NASNetMobile
# 전체 가중치 개수:  1126
# 훈련 가능 개수:  0
# =========================
# 모델명:  NASNetLarge
# 전체 가중치 개수:  1546
# 훈련 가능 개수:  0
# =========================
# 모델명:  EfficientNetB0
# 전체 가중치 개수:  314
# 훈련 가능 개수:  0
# =========================
# 모델명:  EfficientNetB1
# 전체 가중치 개수:  442
# 훈련 가능 개수:  0
# =========================
# 모델명:  EfficientNetB7
# 전체 가중치 개수:  1040
# 훈련 가능 개수:  0
# =========================
# 모델명:  Xception
# 전체 가중치 개수:  236
# 훈련 가능 개수:  0