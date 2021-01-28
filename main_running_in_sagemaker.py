import keras
import wandb
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback
import os
from dotenv import load_dotenv
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import posixpath
from architectures.classification.basic import build_model, build_cnn_model
import random
from keras.optimizers import Adam
# from data.cifar10 import download_data, prepare_data
from data.digit_recognizer import download_data, prepare_data


class LearningRateCallback(keras.callbacks.Callback):
    def __init__(self, model):
        super(LearningRateCallback, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer._decayed_lr('float32').numpy()
        wandb.log({'learning rate': lr})


def train_model(model, x, y, task):
    date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    checkpoints_path = f'/opt/ml/model/checkpoints/{task}/{date}'
    checkpoints_file_format = 'checkpoints.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint_callback = ModelCheckpoint(posixpath.join(checkpoints_path, checkpoints_file_format))

    L = len(x)
    split_ration = 0.8
    train_indexes = random.sample(range(L), int(L * split_ration))
    validation_indexes = list(set(range(L)) - set(train_indexes))

    x_train = x[train_indexes]
    y_train = y[train_indexes]
    x_val = x[validation_indexes]
    y_val = y[validation_indexes]

    generator = ImageDataGenerator(rotation_range=30, zoom_range=0.20,
                                   fill_mode="nearest", shear_range=0.20, horizontal_flip=False,
                                   width_shift_range=0.2, height_shift_range=0.2)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100,
        decay_rate=0.9,
        staircase=False)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])

    model.fit(generator.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) // 32,
              verbose=2, epochs=500, shuffle=True, validation_data=(x_val, y_val),
              callbacks=[WandbCallback(), checkpoint_callback, LearningRateCallback(model)])
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
        wandb.init(project="yanai-test2")

    # task = 'cifar10'
    task = 'digit-recognizer'
    x_train, y_train, x_test, y_test, class_names = download_data()
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    # model = build_model(input_shape, num_classes)
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    train_model(model, x_train, y_train, task)
    test_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
