import tensorflow as tf
from tensorflow import keras


class Generator:
    def __init__(self):
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Dense(7 * 7 * 16, input_dim=128))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Reshape((7, 7, 16)))

        self.model.add(keras.layers.UpSampling2D())
        self.model.add(keras.layers.Conv2D(16, 3, padding='same'))
        self.model.add(keras.layers.LeakyReLU(0.2))

        self.model.add(keras.layers.UpSampling2D())
        self.model.add(keras.layers.Conv2D(8, 3, padding='same'))
        self.model.add(keras.layers.LeakyReLU(0.2))

        self.model.add(keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid'))

    def get_model(self):
        return self.model

# Instantiate the generator model
generator = Generator().get_model()
