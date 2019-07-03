# Ancient Japanese Kana classification with CNN

## Install venv

```shell
$ python -m venv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r ./requirements.txt
```

on Windows, run `<env_name>\Scripts\activate.bat` instead of `source <env_name>/bin/activate`

## Training model

Run

```shell
$ python main.py -d [data_path]
```

For example

```shell
$ python main.py -d /alcon2019/dataset/train_kana
```

As default, all images are read and preprocessed to generate  `./preprocessed_data.npy`
If you modify the model structure and want to retrain again, while the data are the same, you can run

```shell
$ python main.py -d [data_path] -r False
```

to skip the preprocessing. The program will import processed data from `preproccessed_data.npy` directly.

After training, the model is stored in `./model/` and the label binarization object is stored in `./labelbin/`. These two files are used for prediction of new input image.

A visualized result of the loss and accuracy rates over each epoch is stored in `./plot/`.

A json file counting the incorrect validations of each class is stored as `./incorrectCount.json`

## Classify new input image with the trained model

### From terminal

Run

```shell
$ python classify.py -m ./model/model -l ./labelbin/labelbin -i [image_path]
```

to classify a Kana image.

### From code

```python
from classify import KanaClassifier
import cv2

model_path = './model/model'
labelbin_path = './labelbin/labelbin'
image_path = 'some_path'

kc = KanaClassifier(model_path, labelbin_path)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
result = kc.classify(img)
```

`result` is a list of dictionaries containing  two keys, `unicode` and `confidence`, sorted by the confidence from high to low.