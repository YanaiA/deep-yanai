import sagemaker
from os.path import expanduser, join
from sagemaker.tensorflow import TensorFlow
import wandb
import os
import boto3
import posixpath

os.environ['WANDB_DISABLE_CODE'] = '*.patch'
aws_role = os.environ['AWS_IAM_ROLE']

def get_data_to_s3():
    pass


def run_sagemaker():
    experiment_folder = r's3://yanai-temp/SageMaker'
    s3_data_path = posixpath.join(experiment_folder, 'Data', 'cifar10')
    SAGEMAKER_PARAMS = {'machine': 'ml.c4.2xlarge', # 'ml.p3.2xlarge', #'ml.m5.xlarge',
                        'tensorflow_version': '1.15.2', #'2.2', # '1.12',
                        'dir_to_work_on': os.getcwd(),  # all code from this path is copied to container
                        'output_dir': experiment_folder,
                        'checkpoint_s3_dir': posixpath.join(experiment_folder, 'checkpoints')}

    wandb.sagemaker_auth(path="")  # create secret file in this path

    timeout_length = 3600 * 5

    LOCAL = False
    if LOCAL:
        train_instance_type = 'local'
        train_use_spot_instances = False
        checkpoint_s3_uri = None
    else:
        train_instance_type = SAGEMAKER_PARAMS['machine']
        train_use_spot_instances = True
        checkpoint_s3_uri = SAGEMAKER_PARAMS['checkpoint_s3_dir']

    tf_estimator = TensorFlow(entry_point='sagemaker_entry_point.py',
                              role=aws_role,
                              source_dir=SAGEMAKER_PARAMS['dir_to_work_on'],
                              train_instance_count=1,  # is it per role?
                              train_instance_type=train_instance_type,
                              framework_version=SAGEMAKER_PARAMS['tensorflow_version'],
                              py_version='py3',
                              output_path=SAGEMAKER_PARAMS['output_dir'],
                              train_volume_size=100,  # check what is it
                              script_mode=True,  # check
                              hyperparameters=None,  # hyper search
                              train_max_run=timeout_length,
                              checkpoint_s3_uri=checkpoint_s3_uri,
                              train_use_spot_instances=train_use_spot_instances,
                              train_max_wait=timeout_length)  # max time to wait for spot

    tf_estimator.fit({'training': posixpath.join(s3_data_path, 'train'),
                      'validation': posixpath.join(s3_data_path, 'val')})
    # tf_estimator.fit()


def main():
    get_data_to_s3()
    run_sagemaker()


if __name__ == '__main__':
    main()
