#image preprocessing 
def normalize_image(im):
    return im / 255.0

# GaussianBlur : 엣지(edge)와 같이 공간적으로 급변하는 부분 이외의 노이즈를 제거
def GaussianBlur_image(im):
    im_uint8 = (im * 255).astype(np.uint8)
    gray = cv2.cvtColor(im_uint8, cv2.COLOR_RGB2GRAY)
    # filtered_im = median_filter(gray, 5)  # 필터크기 5
    # filtered_im_rgb = cv2.cvtColor(filtered_im, cv2.COLOR_GRAY2RGB)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) ##ksize : 커널 크기 #sigmaX : X 방향 표준편차. 값이 높을수록 이미지가 흐려짐
    img_blur_thresh = cv2.adaptiveThreshold (img_blurred, maxValue=255.0, 
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)
    image_rgb = cv2.cvtColor(img_blur_thresh, cv2.COLOR_GRAY2RGB)

    return image_rgb


def preprocess_image(im):
    im_preprocess = GaussianBlur_image(im)
    im_normalized = normalize_image(im_preprocess)  # 정규화(평균0, 표준편차1)
    return im_normalized


#데이터 
train_dg = ImageDataGenerator(preprocessing_function=preprocess_image)
test_dg = ImageDataGenerator(preprocessing_function=preprocess_image)
val_dg = ImageDataGenerator(preprocessing_function=preprocess_image)