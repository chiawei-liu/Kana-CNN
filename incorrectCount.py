import numpy as np
import os
import cv2
from classify import KanaClassifier
from tqdm import tqdm 
from sklearn.preprocessing import LabelBinarizer

TRAIN_DIR = './test'

count = {}
kc = KanaClassifier('./model/model', './labelbin/labelbin')

for class_dir in tqdm(os.listdir(TRAIN_DIR)): 
    if class_dir[0] is '.':
        continue    

    count[class_dir] = 0   
    
    for img in os.listdir(os.path.join(TRAIN_DIR, class_dir)):
        if img[0] is '.':
            continue
        path = os.path.join(TRAIN_DIR, class_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        result = kc.classify(img)
        if result[0]['unicode'] != class_dir:
            count[class_dir] += 1
    

print(count)