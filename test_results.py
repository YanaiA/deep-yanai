from architectures.classification.basic import build_cnn_model
# from data.cifar10 import download_data, prepare_data
from data.digit_recognizer import download_data, prepare_data
import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def show_results(model, x_test, y_test, class_names):
    for ind in range(10):
        x = x_test[ind,:,:,:]
        y_predicted = model.predict(np.expand_dims(x,0))
        true_class = class_names[np.argmax(y_test[ind,:])]
        predicted_class = class_names[np.argmax(y_predicted)]
        plt.imshow(x)
        plt.title(f'true={true_class}, predicted={predicted_class}')
        plt.show()


def show_cm(model, x_test, y_test, class_names):
    y_predicted = np.argmax(model.predict(x_test), axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    print(confusion_matrix(y_true, y_predicted))


if __name__ == '__main__':
    # task = 'cifar10'
    task = 'digit-recognizer'
    x_train, y_train, x_test, y_test, class_names = download_data()
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    init_model = build_cnn_model(input_shape, num_classes)
    # trained_model = keras.models.load_model(r'trained_models\cifar10_2021-01-07-12-34-49')

    model_to_use = init_model
    # show_cm(model_to_use, x_test, y_test, class_names)
    # show_results(model_to_use, x_test, y_test, class_names)
    show_cm(model_to_use, x_train, y_train, class_names)
    show_results(model_to_use, x_train, y_train, class_names)

    # show_cm(init_model, x_test, y_test, class_names)
    # show_cm(trained_model, x_test, y_test, class_names)
