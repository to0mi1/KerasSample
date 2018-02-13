# -*- coding: utf-8 -*-
import os
from datetime import datetime as dt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import plot_model

# Set Model output dir and filename
if not os.path.exists('./model'):
    os.mkdir('./model')
model_base_dir = './model/'
filename_prefix = dt.now().strftime('%Y%m%d%H%M%S')
filename_suffix = 'type2'
if not os.path.exists('./model/' + filename_prefix + '_' +  filename_suffix):
    os.mkdir('./model/' + filename_prefix + '_' +  filename_suffix)
model_dir = './model/' + filename_prefix + '_' +  filename_suffix

# 分類クラス定義
classes = ['male', 'female','others']
nb_classes = len(classes)


def create_model():
    '''
    Create dnn network model.
    Save the built model in the ./model directory in the json format.
    '''

    print('Start model building')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(150, 150, 3)))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.75))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    model.summary()

    model_json = model.to_json()

    with open(os.path.join(model_dir, filename_prefix + '_' +  filename_suffix + '_model.json'), 'w') as f:
        f.write(model_json)
    plot_model(model, to_file=os.path.join(model_dir, filename_prefix + '_' +  filename_suffix + '_model.png'),
               show_shapes=True)



if __name__ == '__main__':
    create_model()
