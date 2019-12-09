import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                                                                                                                      
import shutil
import argparse
import keras
import numpy as np
from datetime import datetime
import keras_app_models


def main(args):
    # arguments
    num_classes = args.num_classes # default value: 10
    num_epoches = args.num_epoches # default value: 300
    batch_size  = args.batch_size  # default value: 128
    model_arch  = args.model_arch  # default value: "mobilnetv2"
    model_name  = str(model_arch) + '_cifar' + str(num_classes) + '__' + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir     = os.path.join('./log/', model_name) 
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    shutil.copy('./train.py', os.path.join(log_dir, "train.py"))

    # Download Cifar10 dataset
    if num_classes == 10:
        (x_train,y_train), (x_test,y_test) = keras.datasets.cifar10.load_data()
    elif num_classes == 100:
        (x_train,y_train), (x_test,y_test) = keras.datasets.cifar100.load_data()

    # Normalize every pixels: applying min-max normalization
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # convert every class-labels to the one-hot vectors
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.np_utils.to_categorical(y_test, num_classes)

    # build model
    if model_arch == 'mobilenetv2':
        model = keras_app_models.keras_mobilenetv2_cifar(num_classes=num_classes)
    elif model_arch == 'densenet121':
        model = keras_app_models.keras_densenet121_cifar(num_classes=num_classes)
    elif model_arch == 'densenet169':
        model = keras_app_models.keras_densenet169_cifar(num_classes=num_classes)
    elif model_arch == 'densenet201':
        model = keras_densenet201_cifar(num_classes=num_classes)

    model.summary()
    print("trained model is saved in", log_dir)
    
    # callbacks
    callbacks=[]
    callbacks.append(keras.callbacks.CSVLogger(os.path.join(log_dir, 'trainlog.csv'), separator=',', append=False))
    callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'model_checkpoint.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False))

    # compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=num_epoches, 
                        verbose=1, 
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

    # Evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:',score[0])
    print('Test accuracy:',score[1])


if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=10,  type=int, help='number of image classes.')
    parser.add_argument('--num_epoches', default=300, type=int, help='number of training epoches.')
    parser.add_argument('--batch_size',  default=128, type=int, help='batch size.')
    parser.add_argument('--model_arch',  default="mobilenetv2", type=str, choices=['mobilenetv2', 'densenet121', 'densenet169', 'densenet201'], help='model architecture.')
    args = parser.parse_args()
    main(args)
