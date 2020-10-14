from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam


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
