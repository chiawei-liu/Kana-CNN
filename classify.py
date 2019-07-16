
import cv2
import os
import argparse
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import img_to_array

IMG_DIMS = (32, 32, 1)

def confidence(item):
    return item['confidence']

class KanaClassifier:
    def __init__(self, model_path, labelbin_path):
        self.model = load_model(model_path)
        self.lb = pickle.loads(open(labelbin_path, 'rb').read())

    def classify(self, img):
        img = cv2.resize(img, (IMG_DIMS[0], IMG_DIMS[1]))
        ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_OTSU)
        img = img/255
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        proba = self.model.predict(img)[0]
        proba = list(proba)
        for i in range(len(proba)):
            proba[i] = {
                'unicode': self.lb.classes_[i],
                'confidence': proba[i]
            }
        proba.sort(key=confidence, reverse=True)
        return proba

    def classify_orig(self, img):
        img = cv2.resize(img, (IMG_DIMS[0], IMG_DIMS[1]))
        ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_OTSU)
        img = img/255
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return list(self.model.predict(img)[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='path to trained model')
    ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
    ap.add_argument('-i', '--image', required=True, help='path to input image')
    args = vars(ap.parse_args())

    print('loading network...')
    kc = KanaClassifier(args['model'], args['labelbin'])
    img = cv2.imread(args['image'], cv2.IMREAD_GRAYSCALE)
    output = img.copy()

    result = kc.classify(img)
    # print(kc.classify(img))

    label = result[0]['unicode']
    filename = args['image'][(args['image'].rfind(os.path.sep) + 1):]
    print(filename)
    print(result[0]['confidence'])
    correct = 'correct' if filename.rfind(label) != -1 else 'incorrect'

    label = '{}: {:.2f}% ({})'.format(label, result[0]['confidence'] * 100, correct)
    output = cv2.resize(output, (320, 320))
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Output', output)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
    