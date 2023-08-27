import cv2
import numpy as np
from matplotlib import pyplot as plt

# "C:/study_aca/practice/airplane.jfif"

# 컬러로 이미지를 로드합니다.
# image_path = "C:/study_aca/practice/airplane.jfif"


image = cv2.imread("C:/study_aca/practice/airplane.jfif", cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드

# image_path = "C:/study_aca/practice/airplane.jfif"
# image = cv2.imread(image_path)
# image_height, image_width, _ = image.shape

# print("Image height:", image_height)    # 1250
# print("Image width:", image_width)      # 1000


image_10x10 = cv2.resize(image, (10, 10)) # 이미지를 10x10 픽셀 크기로 변환
image_10x10.flatten() # 이미지 데이터를 1차원 벡터로 변환

plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

image_10x10.shape
image_10x10.flatten().shape

#