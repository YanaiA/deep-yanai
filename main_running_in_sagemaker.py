import keras
import wandb
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback
import os
from keras.utils import to_categorical, plot_model
from keras.datasets import cifar10
from dotenv import load_dotenv
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import pickle
import boto3
from botocore.errorfactory import ClientError
import posixpath
from architectures.classification.basic import build_model, build_cnn_model
import random


def download_data(task):
    s3_bucket = 'yanai-temp'
    s3_data_path = 'SageMaker/data'
    file_name = f'data_{task}.p'
    obj_name = posixpath.join(s3_data_path, file_name)

    try:
        boto3.resource('s3').Bucket(s3_bucket).Object(obj_name).download_file(file_name)
        data = pickle.load(open(file_name, 'rb'))
        x_train, y_train, x_test, y_test = data
        print('Data file found in s3')

    except ClientError:  # file not found
        print('Data file not found, downloading')
        if task == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            raise ValueError
        data = [x_train, y_train, x_test, y_test]
        pickle.dump(data, open(file_name, 'wb'))
        print('Data downloaded, uploading to s3')
        boto3.client('s3').upload_file(file_name, s3_bucket, obj_name)
        print('Uploading to s3 finished')

    return x_train, y_train, x_test, y_test


def prepare_data(x_train, y_train, x_test, y_test):
    num_classes = 10

    def _fix_inputs(x, y):
        x = x.astype('float32') / 255.0
        y = to_categorical(y, num_classes)
        return x, y

    (x_train, y_train) = _fix_inputs(x_train, y_train)
    (x_test, y_test) = _fix_inputs(x_test, y_test)
    input_shape = x_train.shape[1:]
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def train_model(model, x, y, task):
    date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    checkpoints_path = f'/opt/ml/model/checkpoints/{task}/{date}'
    checkpoints_file_format = 'checkpoints.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint_callback = ModelCheckpoint(posixpath.join(checkpoints_path, checkpoints_file_format))

    L = len(x)
    split_ration = 0.8
    train_indexes = random.sample(range(L), int(L*split_ration))
    validation_indexes = list(set(range(L)) - set(train_indexes))

    x_train = x[train_indexes]
    y_train = y[train_indexes]
    x_val = x[validation_indexes]
    y_val = y[validation_indexes]

    generator = ImageDataGenerator(rotation_range=30, zoom_range=0.20,
                                   fill_mode="nearest", shear_range=0.20, horizontal_flip=True,
                                   width_shift_range=0.1, height_shift_range=0.1)

    model.fit(generator.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) // 32,
              verbose=2, epochs=100, shuffle=True, validation_data=(x_val, y_val),
              callbacks=[WandbCallback(), checkpoint_callback])
    model.save('/opt/ml/model')


def test_model(model, x_test, y_test):
    model.predict(x_test)


def main():
    print("Running in sagemaker")

    use_wandb = True
    if use_wandb:
        load_dotenv('secrets.env')
        wandb_api_key = os.environ['WANDB_API_KEY']

        os.environ['WANDB_DISABLE_CODE'] = '*.patch'  # do not copy code to wandb
        os.system('wandb login ' + wandb_api_key)
        wandb.init(project="yanai-test1")

    task = 'cifar10'
    x_train, y_train, x_test, y_test = download_data(task)
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    # model = build_model(input_shape, num_classes)
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    train_model(model, x_train, y_train, task)
    test_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
