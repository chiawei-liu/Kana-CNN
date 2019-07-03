import pickle
import os
import numpy as np
import json
from keras.models import load_model
from tqdm import tqdm

def confidence(item):
    return item['confidence']

def countIncorrect(model, lb, image_list, label_list):
    result = model.predict(image_list)

    count = {}
    for clss in lb.classes_:
        count[clss] = 0
    for i in tqdm(range(len(result))):
        proba = result[i]
        proba = list(proba)
        for j in range(len(proba)):
            proba[j] = {
                'label': j,
                'confidence': proba[j]
            }
        proba.sort(key=confidence, reverse=True)
        correctClass = lb.inverse_transform(np.expand_dims(label_list[i], axis=0))[0]
        if lb.classes_[proba[0]['label']] != correctClass:
            count[correctClass] += 1

    return count


def main():
    labelbin_path = './labelbin/labelbin'
    model_path = './model/model'

    lb = pickle.loads(open(labelbin_path, 'rb').read())
    model = load_model(model_path)

    data = np.load('train_data.npy', allow_pickle=True)
    image_list = np.array(list(data[:,0])) # .reshape(-1, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2])
    class_list = np.array(list(data[:,1]))
    label_list = lb.transform(class_list)

    count = countIncorrect(model, lb, image_list, label_list)
    with open('./incorrectCount.json', 'w') as f:
        json.dump(count, f, indent=2)

if __name__ == "__main__":
    main()
