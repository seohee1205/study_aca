#shape 오류인 것 내용 명시하고, 추가 모델 만들기 
#공동 FClayer구성하지 말고 GAP 바로 출력 

#1. VGG19
#2. Xception
#3. ResNet50
#4. ResNet101
#5. InceptionV3
#6. InceptionResNetV2
#7. DenseNet121
#8. MobileNetV2
#9. NASNetMobile
#10. EfficientNetB0

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# model_list = [VGG19, Xception, ResNet50,ResNet101,InceptionV3,
#               InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
#                ]
model_list = [VGG19, ResNet50, ResNet101,
              DenseNet121, MobileNetV2, EfficientNetB0]
# model = VGG16()
# model = VGG19()
#...

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train / 255.
x_test = x_test / 255.

for model in model_list:
    model_instance = model(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model_instance.trainable = True

    # Create a new model
    new_model = Sequential()
    new_model.add(model_instance)
    new_model.add(GlobalAveragePooling2D())
    new_model.add(Dense(100, activation='softmax'))

    # Compile the model
    new_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    new_model.fit(x_train, y_train, batch_size=512, epochs=10, validation_data=(x_test, y_test), verbose=0,
                  callbacks=[EarlyStopping(patience=3), ReduceLROnPlateau(factor=0.2, patience=2)])

    #evaluate
    results = new_model.evaluate(x_test, y_test)
    print("================================")
    print("Model Name:", model.__name__)
    print("Total number of weights:", len(model_instance.weights))
    print("Total number of trainable weights:", len(model_instance.trainable_weights))
    print("loss:", results[0])
    print("acc:", results[1])
    
    
# Model Name: VGG19
# Total number of weights: 32
# Total number of trainable weights: 32
# loss: 2.6189041137695312
# acc: 0.3158999979496002
# ================================
# Model Name: ResNet50
# Total number of weights: 318
# Total number of trainable weights: 212
# loss: 6.058909893035889
# acc: 0.04600000008940697
# ================================
# Model Name: ResNet101
# Total number of weights: 624
# Total number of trainable weights: 416
# loss: 103.00215911865234
# acc: 0.3828999996185303
# ================================
# Model Name: DenseNet121
# Total number of weights: 604
# Total number of trainable weights: 362
# loss: 1.5786384344100952
# acc: 0.6388000249862671
# ================================
# Model Name: MobileNetV2
# Total number of weights: 260
# Total number of trainable weights: 156
# loss: 12.102398872375488
# acc: 0.024299999698996544
# ================================
# Model Name: EfficientNetB0
# Total number of weights: 312
# Total number of trainable weights: 211
# loss: 7.694711685180664
# acc: 0.01600000075995922