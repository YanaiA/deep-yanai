import keras
# from keras.layers import Activation, Dense, GlobalAveragePooling2D, SeparableConv2D, BatchNormalization, concatenate, Lambda
# from keras.models import Model, Input
from keras import layers
import boto3
import wandb
from wandb.keras import WandbCallback
import os
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
import json
from dotenv import load_dotenv

# def get_data_to_s3():
#     S3_path = ''
#     download_data_from_web() # http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
#     split_data()


def prepare_data():
    num_classes = 10

    def _fix_inputs(x, y):
        x = x.astype('float32') / 255.0
        y = to_categorical(y, num_classes)
        return x, y

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # train images = 32*32*3
    # train = 50K, test=10K

    (x_train, y_train) = _fix_inputs(x_train, y_train)
    (x_test, y_test) = _fix_inputs(x_test, y_test)
    input_shape = x_train.shape[1:]
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def build_model(input_shape, num_classes):
    print(input_shape)
    model = Sequential([
        Dense(200, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(150, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=5e-3), metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, verbose=2,
              batch_size=32, epochs=5, shuffle=True, validation_split=0.2,
              callbacks=[WandbCallback()])


def test_model(model, x_test, y_test):
    model.predict(x_test)


def main():
    print("Running in sagemaker")

    WANDB = True
    if WANDB:
        load_dotenv('secrets.env')
        WANDB_API_KEY = os.environ['WANDB_API_KEY']
        print(WANDB_API_KEY)

        os.environ['WANDB_DISABLE_CODE'] = '*.patch'  # do not copy code to wandb
        os.system('wandb login ' + WANDB_API_KEY)
        wandb.init(project="yanai-test1")

    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data()

    # get_data_to_s3()
    model = build_model(input_shape, num_classes)
    train_model(model, x_train, y_train)
    test_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
