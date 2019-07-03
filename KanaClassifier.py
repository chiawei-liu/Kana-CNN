from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import pickle
import cv2

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