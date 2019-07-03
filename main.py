import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import json
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from incorrectCount2 import countIncorrect
from tqdm import tqdm 

TRAIN_DIR = './test'
# TRAIN_DIR = 'G:/computer vision/alcon2019/dataset/train_kana'
IMG_DIMS = (32, 32, 1)
RELOAD_DATA = True
BATCH_SIZE = 32
EPOCHS = 100

ap = argparse.ArgumentParser()
# ap.add_argument('-m', '--model', required=True, help='path to trained model')
# ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
ap.add_argument('-r', '--reprocess', type=bool, default=True, required=False, help='reprocess all images')
ap.add_argument('-d', '--data', required=True, help='path to data set')
args = vars(ap.parse_args())

TRAIN_DIR = args['data']
RELOAD_DATA = args['reprocess']

'''Creating the training data'''
def create_train_data(): 
    image_list = []
    label_list = []
    training_data = []
 
    for class_dir in tqdm(os.listdir(TRAIN_DIR)): 
        if class_dir[0] is '.':
            continue       
        
        for img in tqdm(os.listdir(os.path.join(TRAIN_DIR, class_dir))):
            if img[0] is '.':
                continue
            path = os.path.join(TRAIN_DIR, class_dir, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_DIMS[0], IMG_DIMS[1]))
            ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_OTSU)
            img = img/255
            img = img_to_array(img)
            training_data.append([img, class_dir])

    training_data = np.array(training_data)
    np.random.shuffle(training_data)
    np.save('processed_data.npy', training_data)
    return training_data 


def build_model(width, height, depth, classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(width, height, depth)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    return model

def visualizeResult(history):
    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), history.history['val_loss'], label='val_lose')
    plt.plot(np.arange(0, N), history.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), history.history['val_acc'], label='val_acc')
    plt.plot(np.arange(0, N), history.history['loss'], label='train_loss')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper left')
    plt.savefig('./plot/result.png')


training_data = create_train_data() if RELOAD_DATA else np.load('train_data.npy', allow_pickle=True)

# training_data = np.load('train_data.npy', allow_pickle=True)
image_list = np.array(list(training_data[:,0])) # .reshape(-1, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2])
class_list = np.array(list(training_data[:,1]))

lb = LabelBinarizer()
label_list = lb.fit_transform(class_list)

(trainX, testX, trainY, testY) = train_test_split(image_list, label_list, test_size=0.2)
model = build_model(IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2], len(lb.classes_))
print('compiling model')
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('training model')
history = model.fit(x=trainX, y=trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(testX, testY))

print('serializing network')
model.save('./model/model')

print('serializing label binarizer')
f = open('./labelbin/labelbin', 'wb')
f.write(pickle.dumps(lb))
f.close()

count = countIncorrect(model, lb, testX, testY)
with open('./incorrectCount.json', 'w') as f:
    json.dump(count, f, indent=2)

visualizeResult(history)

'''
plt.figure()
plt.imshow(image_list[0])
plt.colorbar()
plt.grid(False)
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_list[i], cmap=plt.cm.binary)
    plt.xlabel(label_list[i])
plt.show()


print("shape of datas: {}\tshape of labels: {}".format(image_list.shape, 
label_list.shape))
print(image_list)
print(len(label_list))
print(class_names)
print(label_list)

'''

