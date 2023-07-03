from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense


class Discriminator:
    def __init__(self):
        self.model = Sequential()

        # First Conv Block
        self.model.add(Conv2D(32, 5, input_shape=(256, 256, 1)))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.4))

        # Second Conv Block
        self.model.add(Conv2D(64, 5))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.4))

        # Third Conv Block
        self.model.add(Conv2D(128, 5))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.4))

        # Fourth Conv Block
        self.model.add(Conv2D(256, 5))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.4))

        # Flatten then pass to dense layer
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation='sigmoid'))

    def get_model(self):
        return self.model
