import keras
import wandb
from wandb.keras import WandbCallback
import os
from keras.utils import to_categorical
from keras.datasets import cifar10
from dotenv import load_dotenv
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import pickle
import boto3
from botocore.errorfactory import ClientError
import posixpath
from architectures.classification.basic import build_model


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


def train_model(model, x_train, y_train, task):
    date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    checkpoints_path = f'/opt/ml/model/checkpoints/{task}/{date}'
    checkpoints_file_format = 'checkpoints.{epoch:02d}-{val_loss:.2f}.hdf5'
    model.fit(x_train, y_train, verbose=2,
              batch_size=32, epochs=2, shuffle=True, validation_split=0.2,
              callbacks=[WandbCallback(),
                         ModelCheckpoint(posixpath.join(checkpoints_path, checkpoints_file_format))])
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

    model = build_model(input_shape, num_classes)
    train_model(model, x_train, y_train, task)
    test_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
