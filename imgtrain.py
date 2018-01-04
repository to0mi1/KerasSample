# -*- coding: utf-8 -*-
import os
import argparse
import csv
from datetime import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger


# ディレクトリの初期化 (トレーニング毎に結果ディレクトリ配下に新規ディレクトリを作成する)
filename_prefix = dt.now().strftime('%Y%m%d%H%M%S')
result_dir = './results/' + filename_prefix
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# トレーニングデータ、バリデーションデータ共に以下のディレクトリへ画像を格納する
train_path = "./data/train"
validation_path = "./data/valid"

# 分類クラス定義
classes = ['male', 'female']
nb_classes = len(classes)

# ハイパーパラメータ
batch_size = 16
nb_epoch = 30
optimizer = 'RMSprop'
activation = 'relu'
loss = 'binary_crossentropy'
dropout_rate = [0.0, 0.0, 0.0]
image_width = 80
image_height = 80
ajustimage = True


def create_model():

    print('Start model building')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(image_width, image_height, 3)))
    model.add(Activation(activation))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate[0]))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate[1]))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate[2]))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    print('\n')

    model_json = model.to_json()
    with open(os.path.join(result_dir, filename_prefix +'_model.json'), 'w') as f:
        f.write(model_json)
    print('Model saved in %s'% (os.path.join(result_dir, filename_prefix +'_model.json')))
    print('Build complete!!')

    return model

def train(model):
    print('Start training.')

    # トレーニング用データの生成
    if ajustimage :
        print('Randomly adjust to generate training data.')
        train_datagen = ImageDataGenerator(
            #zca_whitening= True, # ZCA白色化を適用します
            rotation_range = 40, # 画像をランダムに回転する回転範囲
            width_shift_range = 0.2, # ランダムに水平シフトする範囲
            height_shift_range = 0.2, # ランダムに垂直シフトする範囲
            shear_range = 0.2, # シアー強度（反時計回りのシアー角度（ラジアン））
            zoom_range = 0.2, # ランダムにズームする範囲．浮動小数点数が与えられた場合
            horizontal_flip = True, # 水平方向に入力をランダムに反転します
            rescale = 1.0 / 255) # Noneか0ならば，適用しない．それ以外であれば，(他の変換を行う前に) 与えられた値をデータに積算する
    else :
        print('Training data is generated with the raw data.')
        train_datagen = ImageDataGenerator()

    # 検証用データの生成
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = (image_width, image_height),
        batch_size = batch_size,
        classes = classes,
        class_mode = 'categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(image_width, image_height),
        batch_size= batch_size,
        classes=classes,
        class_mode='categorical')
    
    print(train_generator.class_indices)

    steps_per_epoch = train_generator.samples
    validation_steps = validation_generator.samples

    print('steps_per_epoch is set to %s' % steps_per_epoch)
    print('validation_steps is set to %s' % validation_steps)

    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    csv_logger = CSVLogger(os.path.join(result_dir, filename_prefix +'_log.csv'), separator=',')

    print('Training will start')
    print('\n')

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=2,
                                  callbacks=[es_cb, csv_logger],
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  epochs=nb_epoch)


    print('Training Complete.')
    print('\n')

    model.save_weights(os.path.join(result_dir, filename_prefix + '_weight.h5'))
    plot_model(model, to_file=os.path.join(result_dir, filename_prefix + '_model.png'), show_shapes=True)

    print('\n')
    print('last epoch loss:%f, acc:%f, val_loss:%f, val_acc:%f' % (
        history.history['loss'][-1],
        history.history['acc'][-1],
        history.history['val_loss'][-1],
        history.history['val_acc'][-1] ))
    return history


def main() :
    print('Processing starts.')
    print('Result is saved in %s' % (os.path.join(result_dir)))

    print('\n')

    model = create_model()

    history = train(model)

    csv_row = [result_dir, classes, optimizer, batch_size, nb_epoch, dropout_rate, image_width, image_height,
                 history.history['loss'][-1], history.history['acc'][-1], 
                 history.history['val_loss'][-1], history.history['val_acc'][-1]]

    with open('./results/summary.csv', 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC) 
        writer.writerow(csv_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', help='分類クラスのリスト. 例) "male,female"', default='male,female')
    parser.add_argument('--targetsize-width', type=int, dest='target_width', help='変換先画像サイズの幅', metavar='shape_width', default=150)
    parser.add_argument('--targetsize-height', type=int, dest='target_height', help='変換先画像サイズの高さ', metavar='shape_height', default=150)
    parser.add_argument('--dropout1', type=float, help='ドロップアウト', default=0.25)
    parser.add_argument('--dropout2', type=float, help='ドロップアウト', default=0.25)
    parser.add_argument('--dropout3', type=float, help='ドロップアウト', default=0.5)
    parser.add_argument('--batchsize', type=int, help='バッチサイズ', default=16)
    parser.add_argument('--optimizer', help='最適化関数', default='RMSprop')
    parser.add_argument('--epochs', type=int, help='エポック数', default=30)
    parser.add_argument('--ajustimage', type=bool, help='ランダムに画像を変異させる', default=True)
    args = parser.parse_args()

    classes = args.classes.split(',')
    nb_classes= len(classes)

    image_width = args.target_width
    image_height = args.target_height
    dropout_rate = [args.dropout1, args.dropout2, args.dropout3]
    batch_size = args.batchsize
    nb_epoch = args.epochs
    optimizer = args.optimizer

    print('Classify into %d classes of %s' % (nb_classes, classes))
    print('Image will be resizing to %dx%d' % (image_width, image_height ))
    print('Rate of dropout is %.2f, %.2f, %.2f' % (dropout_rate[0], dropout_rate[1], dropout_rate[2]))
    print('Batchsize is set to %d' % (batch_size))
    print('Epochs is set to %d' % (nb_epoch))
    print('Optimizer is set to %s' % (optimizer))
    print('Random adjustment of image is set to %s' % (ajustimage))

    print('\n')

    main()
