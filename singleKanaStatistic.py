import numpy as np
import os
import cv2
import csv
import argparse
from classify import KanaClassifier
from tqdm import tqdm 
from sklearn.preprocessing import LabelBinarizer


def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='path to trained model')
    ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
    ap.add_argument('-i', '--input', required=True, help='path to input images')
    ap.add_argument('-o', '--output', required=False, help='path to output folder', default='./singleKanaStatistic')
    args = vars(ap.parse_args())


    if not os.path.exists(args['output']):
        os.makedirs(args['output'])

    kc = KanaClassifier(args['model'], args['labelbin'])

    fieldnames = [None] + list(kc.lb.classes_)
    # confidence = {}

    for class_dir in tqdm(os.listdir(args['input'])): 
        if class_dir[0] is '.':
            continue

        with open(os.path.join(args['output'], class_dir + '.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fieldnames)

            for img_name in os.listdir(os.path.join(args['input'], class_dir)):
                if img_name[0] is '.':
                    continue
                path = os.path.join(args['input'], class_dir, img_name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                result = kc.classify_orig(img)
                
                # for i in range(len(result)):
                #    confidence[kc.lb.classes_[i]] = result[i]
                
                writer.writerow([img_name] + result)
                

if __name__ == "__main__":
    main()