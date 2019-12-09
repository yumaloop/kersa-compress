import keras
from keras_app_models import keras_mobilnetv2_cifar

mobilenetv2_cifar10 = keras_mobilnetv2_cifar(num_classes=10)
mobilenetv2_cifar10.summary()


