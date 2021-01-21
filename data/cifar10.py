import pickle
import boto3
from botocore.errorfactory import ClientError
from keras.utils import to_categorical
from keras.datasets import cifar10
import posixpath
import os
import numpy as np


def download_data(task):
    s3_bucket = 'yanai-temp'
    s3_data_path = 'SageMaker/data'
    file_name = f'data_{task}.p'
    obj_name = posixpath.join(s3_data_path, file_name)

    try:
        if not os.path.isfile(file_name):
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

    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return x_train, y_train, x_test, y_test, class_names


def prepare_data(x_train, y_train, x_test, y_test):
    num_classes = np.max(y_train)

    def _fix_inputs(x, y):
        x = x.astype('float32') / 255.0
        y = to_categorical(y, num_classes+1)
        return x, y

    (x_train, y_train) = _fix_inputs(x_train, y_train)
    (x_test, y_test) = _fix_inputs(x_test, y_test)
    input_shape = x_train.shape[1:]
    return x_train, y_train, x_test, y_test, input_shape, num_classes
