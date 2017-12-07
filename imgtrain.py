# -*- coding: utf-8 -*-

import os
import numpy as np
from datetime import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

result_dir = './results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
train_path = "./data/train"
validation_path = "./data/valid"
filename_suffix = dt.now().strftime('%Y%m%d%H%M%S')

# 分類クラス
classes = ['male', 'female']
nb_classes = len(classes)
batch_size = 16
nb_epoch = 30
optimizer = 'RMSprop'
loss = 'binary_crossentropy'


def create_model():
    logger.info('Build model......')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    model_json = model.to_json()
    with open(os.path.join(result_dir, 'model' + filename_suffix +'.json'), 'w') as f:
        f.write(model_json)

    logger.info('Build complete!!')

    return model


def train(model):
    train_datagen = ImageDataGenerator(
        #zca_whitening= True, # ZCA白色化を適用します
        rotation_range = 40, # 画像をランダムに回転する回転範囲
        width_shift_range = 0.2, # ランダムに水平シフトする範囲
        height_shift_range = 0.2, # ランダムに垂直シフトする範囲
        shear_range = 0.2, # シアー強度（反時計回りのシアー角度（ラジアン））
        zoom_range = 0.2, # ランダムにズームする範囲．浮動小数点数が与えられた場合
        horizontal_flip = True, # 水平方向に入力をランダムに反転します
        rescale = 1.0 / 255) # Noneか0ならば，適用しない．それ以外であれば，(他の変換を行う前に) 与えられた値をデータに積算する


    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    logger.info('batch_size is set to %d',batch_size)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = (150, 150),
        batch_size = batch_size,
        classes = classes,
        class_mode = 'categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(150, 150),
        batch_size= batch_size,
        classes=classes,
        class_mode='categorical')

    steps_per_epoch = train_generator.samples
    validation_steps = validation_generator.samples

    logger.info('steps_per_epoch is set to %s', steps_per_epoch)
    logger.info('validation_steps is set to %s', validation_steps)

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    logger.info("Set callback. mode is auto.")

    logger.debug('Training will start...')
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=[es_cb],
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  epochs=nb_epoch)
    logger.debug('Train complete')

    model.save_weights(os.path.join(result_dir, 'weight' + filename_suffix + '.h5'))
    #plot_model(model, to_file=os.path.join(result_dir, 'model' + filename_suffix + '.png'), show_shapes=True)


def main():
    model = create_model()
    train(model)


if __name__ == '__main__':
    main()
