# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from datetime import datetime as dt
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

base_filename = "20171206110000"
result_dir = './results'

classes = ['male', 'female']

def load_model():
    logger.info('Load model......')

    model_json = open(os.path.join(result_dir, 'model' + base_filename +'.json')).read()
    model = model_from_json(model_json)
    model.load_weights(os.path.join(result_dir, 'weight' + base_filename +'.h5'))
    return model

def predict(model, path):
    img = image.load_img(path, target_size=(150, 150))

    logger.info('classify image of %s', path)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x)[0]

    top = 2
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    for x in result:
        print(x)

def main(args):
    model = load_model()
    predict(model, args.image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", "-i", help="")
    args = parser.parse_args()
    main(args)
