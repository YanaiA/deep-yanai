import os, argparse
import subprocess
import sys
import boto3


def install_packages():
    OPEN_CV = False
    if OPEN_CV:
        os.system('apt-get update')
        os.system('apt-get install -y libsm6 libxext6 libxrender-dev --allow-unauthenticated')

    # TODO check if there is a build it option to install requirements
    subprocess.call([sys.executable, "-m", "pip", "install", "-U", "-r", '/opt/ml/code/requirements.txt'])

    ALGO_INFRA = False
    if ALGO_INFRA:
        ALGO_INFRA_PACKAGE_BUCKET = 'nanit-algo-resources'
        ALGO_INFRA_PACKAGE_REL_PATH = 'algo-infra/nanit_algo_infra-0.1.6-py3-none-any.whl'
        s3_client = boto3.client('s3')
        algo_infra_dest_path = '/opt/ml/code/nanit_algo_infra-0.1.2-py3-none-any.whl'
        s3_client.download_file(ALGO_INFRA_PACKAGE_BUCKET, ALGO_INFRA_PACKAGE_REL_PATH, algo_infra_dest_path)
        subprocess.call(
            [sys.executable, "-m", "pip", "install", '/opt/ml/code/nanit_algo_infra-0.1.2-py3-none-any.whl'])


if __name__ == '__main__':
    install_packages()
    parser = argparse.ArgumentParser()

    # parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()

    from main_running_in_sagemaker import *
    main()
