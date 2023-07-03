from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, UpSampling2D


class Generator:
    def __init__(self):
        self.model = Sequential()

        self.model.add(Dense(7 * 7 * 16, input_dim=128))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Reshape((7, 7, 16)))

        self.model.add(UpSampling2D())
        self.model.add(Conv2D(16, 3, padding='same'))
        self.model.add(LeakyReLU(0.2))

        self.model.add(UpSampling2D())
        self.model.add(Conv2D(8, 3, padding='same'))
        self.model.add(LeakyReLU(0.2))

        self.model.add(Conv2D(1, 3, padding='same', activation='sigmoid'))

    def get_model(self):
        return self.model
