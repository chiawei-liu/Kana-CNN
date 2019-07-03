import os
import shutil
from tqdm import tqdm

DEST_DIR = './test'
SOURCE_DIR = 'G:/computer vision/alcon2019/dataset/train_kana'

for class_dir in tqdm(os.listdir(SOURCE_DIR)): 
    if class_dir[0] is '.':
        continue
    
    dest_class_dir = os.path.join(DEST_DIR, class_dir)
    os.mkdir(dest_class_dir)
    
    source_class_dir = os.path.join(SOURCE_DIR, class_dir)
    i = 0
    for img in tqdm(os.listdir(source_class_dir)):
        if img[0] is '.':
            continue
        i += 1
        if i > 250:
            break

        img_path = os.path.join(source_class_dir, img)
        shutil.copy2(img_path, dest_class_dir)
        