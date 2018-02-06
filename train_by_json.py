'''train'''
# -*- coding: utf-8 -*-
import os
from datetime import datetime as dt
import argparse
import csv
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger

train_path = "./data/train"
validation_path = "./data/valid"

filename_prefix = dt.now().strftime('%Y%m%d%H%M%S')
if not os.path.exists('./results'):
    os.mkdir('./results')
result_dir = './results/' + filename_prefix
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def train(model, optimizer='Adam', classes=['1', '2'], targetsize_width=150, targetsize_height=150, batch_size=16, epochs=30, imagedatagen_args={'rescale' : 1 / 255}):
    # モデルの読み込み
    print('load the model of ' + model)
    print('set the optimizer to ' + optimizer)
    model_json = open(os.path.join(model)).read()
    model = model_from_json(model_json)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    # トレーニングデータの前処理
    print('Preprocessing training data')
    train_datagen = ImageDataGenerator(**imagedatagen_args)
    print('rescale : ' + str(train_datagen.rescale))
    print('rotation_range : ' + str(train_datagen.rotation_range))
    print('width_shift_range : ' + str(train_datagen.width_shift_range))
    print('height_shift_range : ' + str(train_datagen.height_shift_range))
    print('shear_range :' + str(train_datagen.shear_range))
    print('zoom_range :' + str(train_datagen.zoom_range))
    print('horizontal_flip :' + str(train_datagen.horizontal_flip))
    print('vertical_flip :' + str(train_datagen.vertical_flip))

    # 検証用データの生成
    print('Preprocessing validation data')
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # トレーニングデータの生成
    print('Generation of training data')
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(targetsize_width, targetsize_height),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')
    print('target size : ' + str(targetsize_width) + ' x ' + str(targetsize_height))
    print('batch_size:' + str(batch_size))
    print(train_generator.class_indices)

    # 検証用データの生成
    print('Generation of validation data')
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(targetsize_width, targetsize_width),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    print('Training will start')
    steps_per_epoch = train_generator.samples
    print('steps_per_epoch is set to %s' % steps_per_epoch)
    validation_steps = validation_generator.samples
    print('validation_steps is set to %s' % validation_steps)
    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    csv_logger = CSVLogger(os.path.join(result_dir, filename_prefix +'_log.csv'), separator=',')

    history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=steps_per_epoch,
                                verbose=2,
                                callbacks=[es_cb, csv_logger],
                                validation_data=validation_generator,
                                validation_steps=validation_steps,
                                epochs=epochs)
    print('Training Complete.')
    
    model.save_weights(os.path.join(result_dir, filename_prefix + '_weight.h5'))

    print('last epoch loss:%f, acc:%f, val_loss:%f, val_acc:%f' % (
        history.history['loss'][-1],
        history.history['acc'][-1],
        history.history['val_loss'][-1],
        history.history['val_acc'][-1]))
    
    csv_row = [model, optimizer, classes, targetsize_width, targetsize_height,
               batch_size, epochs, imagedatagen_args,
               history.history['loss'][-1], history.history['acc'][-1],
               history.history['val_loss'][-1], history.history['val_acc'][-1]]

    with open('./results/summary.csv', 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(csv_row)
    print('The result was added to ./results/summary.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='モデルのパス')
    parser.add_argument('--classes', help='分類クラスのリスト. 例) "male,female"', default='male,female')
    parser.add_argument('--optimizer', help='optimizer', default='Adam')
    parser.add_argument('--targetsize-width', type=int, dest='target_width', help='変換先画像サイズの幅', metavar='shape_width', default=150)
    parser.add_argument('--targetsize-height', type=int, dest='target_height', help='変換先画像サイズの高さ', metavar='shape_height', default=150)
    parser.add_argument('--batchsize', type=int, help='バッチサイズ', default=16)
    parser.add_argument('--epochs', type=int, help='エポック数', default=30)
    parser.add_argument('--rotation-range', type=float, dest='rotation_range', help='画像をランダムに回転する回転範囲', default=0.)
    parser.add_argument('--width-shift-range', type=float, dest='width_shift_range', help='ランダムに水平シフトする範囲', default=0.)
    parser.add_argument('--height-shift-range', type=float, dest='height_shift_range', help='ランダムに垂直シフトする範囲', default=0.)
    parser.add_argument('--shear-range', type=float, dest='shear_range', help='シアー強度（反時計回りのシアー角度（ラジアン））', default=0.)
    parser.add_argument('--zoom-range', type=float, dest='zoom_range', help='ランダムにズームする範囲．浮動小数点数が与えられた場合', default=0.)
    parser.add_argument('--horizontal-flip', dest='horizontal_flip', action='store_true', help='水平方向に入力をランダムに反転します', default=False)
    parser.add_argument('--vertical-flip', dest='vertical_flip', action='store_true', help='垂直方向に入力をランダムに反転します', default=False)
    args = parser.parse_args()

    classes = args.classes.split(',')

    imagedatagen_args = {
        'rescale' : 1 / 255,
        'rotation_range': args.rotation_range,
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
        'horizontal_flip': args.horizontal_flip,
        'vertical_flip': args.vertical_flip
    }

    train(
        args.model,
        classes=classes,
        optimizer=args.optimizer,
        targetsize_width=args.target_width,
        targetsize_height=args.target_height,
        batch_size=args.batchsize,
        epochs=args.epochs,
        imagedatagen_args=imagedatagen_args
    )