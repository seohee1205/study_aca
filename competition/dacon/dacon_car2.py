# %pip install ultralytics
# %pip install scikit-learn

import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

SEED = 42
BATCH_SIZE = 8
MODEL = "v2"

if os.path.exists("../data/yolo"):
    shutil.rmtree("../data/yolo")

if not os.path.exists("../data/yolo/train"):
    os.makedirs("../data/yolo/train")
    
if not os.path.exists("../data/yolo/valid"):
    os.makedirs("../data/yolo/valid")
    
if not os.path.exists("../data/yolo/test"):
    os.makedirs("../data/yolo/test")    
    
if not os.path.exists("../results"):
    os.makedirs("../results")

def make_yolo_dataset(image_paths, txt_paths, type="train"):
    for image_path, txt_path in tqdm(zip(image_paths, txt_paths if not type == "test" else image_paths), total=len(image_paths)):
        source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)        
        image_height, image_width, _ = source_image.shape
        
        target_image_path = f"../data/yolo/{type}/{os.path.basename(image_path)}"
        cv2.imwrite(target_image_path, source_image)
        
        if type == "test":
            continue
        
        with open(txt_path, "r") as reader:
            yolo_labels = []
            for line in reader.readlines():
                line = list(map(float, line.strip().split(" ")))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x, y = float(((x_min + x_max) / 2) / image_width), float(((y_min + y_max) / 2) / image_height)
                w, h = abs(x_max - x_min) / image_width, abs(y_max - y_min) / image_height
                yolo_labels.append(f"{class_name} {x} {y} {w} {h}")
            
        target_label_txt = f"../data/yolo/{type}/{os.path.basename(txt_path)}"      
        with open(target_label_txt, "w") as writer:
            for yolo_label in yolo_labels:
                writer.write(f"{yolo_label}\n")

image_paths = sorted(glob("../data/train/*.png"))
txt_paths = sorted(glob("../data/train/*.txt"))

train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths = train_test_split(image_paths, txt_paths, test_size=0.1, random_state=SEED)

make_yolo_dataset(train_images_paths, train_txt_paths, "train")
make_yolo_dataset(valid_images_paths, valid_txt_paths, "valid")
make_yolo_dataset(sorted(glob("../data/test/*.png")), None, "test")

with open("../data/classes.txt", "r") as reader:
    lines = reader.readlines()
    classes = [line.strip().split(",")[1] for line in lines]

yaml_data = {
              "names": classes,
              "nc": len(classes),
              "path": "/Data/데이터 분석 대회/DACON/합성데이터 기반 객체 탐지 AI 경진대회/data/yolo/",
              "train": "train",
              "val": "valid",
              "test": "test"
            }

with open("../data/yolocustom.yaml", "w") as writer:
    yaml.dump(yaml_data, writer)

#model = YOLO(f"{MODEL}/train/weights/last.pt")
model = YOLO("yolov8x")
results = model.train(
    data="../data/yolo/custom.yaml",
    imgsz=(1024, 1024),
    epochs=200,
    batch=BATCH_SIZE,
    patience=5,
    workers=16,
    device=0,
    exist_ok=True,    
    project=f"{MODEL}",
    name="train",
    seed=SEED,
    pretrained=False,
    resume=True,
    optimizer="Adam",
    lr0=1e-3,
    augment=True,
    val=True,
    cache=True
    )

def get_test_image_paths(test_image_paths):    
    for i in range(0, len(test_image_paths), BATCH_SIZE):
        yield test_image_paths[i:i+BATCH_SIZE]

model = YOLO("v2/train/weights/best.pt")
test_image_paths = glob("../data/yolo/test/*.png")
for i, image in tqdm(enumerate(get_test_image_paths(test_image_paths)), total=int(len(test_image_paths)/BATCH_SIZE)):
    model.predict(image, imgsz=(1024, 1024), iou=0.2, conf=0.5, save_conf=True, save=False, save_txt=True, project=f"{MODEL}", name="predict",
                  exist_ok=True, device=0, augment=True, verbose=False)
    if i % 5 == 0:
        clear_output(wait=True)

def yolo_to_labelme(line, image_width, image_height, txt_file_name):    
    file_name = txt_file_name.split("/")[-1].replace(".txt", ".png")
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]
    
    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)
    
    return file_name, int(class_id), confidence, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max

infer_txts = glob(f"{MODEL}/predict/labels/*.txt")

results = []
for infer_txt in tqdm(infer_txts):
    base_file_name = infer_txt.split("/")[-1].split(".")[0]
    imgage_height, imgage_width = cv2.imread(f"../data/test/{base_file_name}.png").shape[:2]        
    with open(infer_txt, "r") as reader:        
        lines = reader.readlines()        
        for line in lines:
            results.append(yolo_to_labelme(line, imgage_width, imgage_height, infer_txt))

df_submission = pd.DataFrame(data=results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
df_submission.to_csv(f"../results/{MODEL}.csv", index=False)