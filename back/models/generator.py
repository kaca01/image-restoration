from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, UpSampling2D


class Generator:
    def __init__(self):
        self.model = Sequential()

        # Takes in random values and reshapes it to 7x7x128
        # Beginnings of a generated image
        self.model.add(Dense(7 * 7 * 128, input_dim=128))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Reshape((7, 7, 128)))

        # Upsampling block 1
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(128, 5, padding='same'))
        self.model.add(LeakyReLU(0.2))

        # Upsampling block 2
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(128, 5, padding='same'))
        self.model.add(LeakyReLU(0.2))

        # Convolutional block 1
        self.model.add(Conv2D(128, 4, padding='same'))
        self.model.add(LeakyReLU(0.2))

        # Convolutional block 2
        self.model.add(Conv2D(128, 4, padding='same'))
        self.model.add(LeakyReLU(0.2))

        # Conv layer to get to one channel
        self.model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    def get_model(self):
        return self.model
