from keras.models import Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from tensorflow.python.keras.models import Model


def build_model(input_shape, num_classes):
    print(input_shape)
    model = Sequential([
        Dense(200, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(150, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_cnn_model(inputs_shape, num_classes):
    def add_one_group(filters, strides):
        group = Sequential([
            Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same'),
            BatchNormalization(),
            LeakyReLU()
        ])
        return group

    input_layer = Input(inputs_shape)
    x1 = add_one_group(32, 1)(input_layer)
    x2 = add_one_group(32, 2)(x1)
    x3 = add_one_group(64, 1)(x2)
    x4 = add_one_group(64, 2)(x3)

    x5 = Flatten()(x4)

    x6 = Dense(128)(x5)
    x7 = BatchNormalization()(x6)
    x8 = LeakyReLU()(x7)
    x9 = Dropout(rate=0.5)(x8)

    x10 = Dense(num_classes)(x9)
    output_layer = Activation('softmax')(x10)

    model = Model(input_layer, output_layer)
    return model
