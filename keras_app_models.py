import keras

def keras_mobilenetv2_cifar(input_shape=(32,32,3), num_classes=10):
    # dataformat must be 'channel_last': (height, width, channels)
    # see https://keras.io/ja/applications/#mobilenetv2
    input_tensor = keras.layers.Input(shape=input_shape)
    mobilenetv2_imagenet = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = keras.models.Sequential()
    top_model.add(keras.layers.Flatten(input_shape=mobilenetv2_imagenet.output_shape[1:]))
    top_model.add(keras.layers.Dense(256))
    top_model.add(keras.layers.Activation('relu'))
    top_model.add(keras.layers.BatchNormalization())
    top_model.add(keras.layers.Dropout(0.4))
    top_model.add(keras.layers.Dense(num_classes))
    top_model.add(keras.layers.Activation('relu'))

    mobilnetv2_cifar = keras.models.Model(inputs=mobilenetv2_imagenet.input, outputs=top_model(mobilenetv2_imagenet.output))
    return mobilenetv2_cifar


def keras_densenet121_cifar(input_shape=(32,32,3), num_classes=10):
    # dataformat must be 'channel_last': (height, width, channels)i
    # see https://keras.io/ja/applications/#densenet
    input_tensor = keras.layers.Input(shape=input_shape)
    densenet121_imagenet = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = keras.models.Sequential()
    top_model.add(keras.layers.Flatten(input_shape=mobilenetv2_imagenet.output_shape[1:]))
    top_model.add(keras.layers.Dense(256))
    top_model.add(keras.layers.Activation('relu'))
    top_model.add(keras.layers.BatchNormalization())
    top_model.add(keras.layers.Dropout(0.4))
    top_model.add(keras.layers.Dense(num_classes))
    top_model.add(keras.layers.Activation('relu'))

    densenet121_cifar = keras.models.Model(inputs=densenet121_imagenet.input, outputs=top_model(densenet121_imagenet.output))
    return densenet121_cifar


def keras_densenet169_cifar(input_shape=(32,32,3), num_classes=10):
    # dataformat must be 'channel_last': (height, width, channels)i
    # see https://keras.io/ja/applications/#densenet
    input_tensor = keras.layers.Input(shape=input_shape)
    densenet169_imagenet = keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = keras.models.Sequential()
    top_model.add(keras.layers.Flatten(input_shape=mobilenetv2_imagenet.output_shape[1:]))
    top_model.add(keras.layers.Dense(256))
    top_model.add(keras.layers.Activation('relu'))
    top_model.add(keras.layers.BatchNormalization())
    top_model.add(keras.layers.Dropout(0.4))
    top_model.add(keras.layers.Dense(num_classes))
    top_model.add(keras.layers.Activation('relu'))

    densenet169_cifar = keras.models.Model(inputs=densenet169_imagenet.input, outputs=top_model(densenet169_imagenet.output))
    return densenet169_cifar


def keras_densenet201_cifar(input_shape=(32,32,3), num_classes=10):
    # dataformat must be 'channel_last': (height, width, channels)i
    # see https://keras.io/ja/applications/#densenet
    input_tensor = keras.layers.Input(shape=input_shape)
    densenet201_imagenet = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = keras.models.Sequential()
    top_model.add(keras.layers.Flatten(input_shape=mobilenetv2_imagenet.output_shape[1:]))
    top_model.add(keras.layers.Dense(256))
    top_model.add(keras.layers.Activation('relu'))
    top_model.add(keras.layers.BatchNormalization())
    top_model.add(keras.layers.Dropout(0.4))
    top_model.add(keras.layers.Dense(num_classes))
    top_model.add(keras.layers.Activation('relu'))

    densenet201_cifar = keras.models.Model(inputs=densenet201_imagenet.input, outputs=top_model(densenet201_imagenet.output))
    return densenet201_cifar
