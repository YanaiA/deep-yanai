import pickle
import boto3
from botocore.errorfactory import ClientError
from keras.utils import to_categorical
import posixpath
import os
import numpy as np
import csv


def download_data():
    s3_bucket = 'yanai-temp'
    s3_data_path = 'SageMaker/data'
    file_name = f'data_digit_recognizer.p'
    obj_name = posixpath.join(s3_data_path, file_name)

    try:
        if not os.path.isfile(file_name):
            boto3.resource('s3').Bucket(s3_bucket).Object(obj_name).download_file(file_name)

        data = pickle.load(open(file_name, 'rb'))
        x_train, y_train, x_test, y_test = data
        print('Data file found in s3')

    except ClientError:  # file not found
        print('Data file not found, downloading')

        def load_file(filename, is_train):
            x = []
            y = []
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                next(reader) # skip header
                for row in reader:
                    if is_train:
                        y.append(int(row[0]))
                        x.append([int(val) for val in row[1:]])
                    else:
                        y.append(-1)
                        x.append([int(val) for val in row])
            return x,y

        x_train, y_train = load_file(r'data\digit-recognizer\train.csv', True)
        x_test, y_test = load_file(r'data\digit-recognizer\test.csv', False)

        data = [x_train, y_train, x_test, y_test]
        pickle.dump(data, open(file_name, 'wb'))
        print('Data downloaded, uploading to s3')
        boto3.client('s3').upload_file(file_name, s3_bucket, obj_name)
        print('Uploading to s3 finished')

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return x_train, y_train, x_test, y_test, class_names


def prepare_data(x_train, y_train, x_test, y_test):
    num_classes = np.max(y_train) - np.min(y_train) + 1

    def _fix_inputs(x, y):
        x = np.reshape(np.array(x).astype('float32') / 255.0, (len(y), 28, 28, 1))
        y = to_categorical(y, num_classes)
        return x, y

    (x_train, y_train) = _fix_inputs(x_train, y_train)
    (x_test, y_test) = _fix_inputs(x_test, y_test)
    input_shape = x_train.shape[1:]
    return x_train, y_train, x_test, y_test, input_shape, num_classes
