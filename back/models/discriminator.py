import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense


class Discriminator:
    def __init__(self):
        self.model = keras.models.Sequential()

        # First Conv Block
        self.model.add(keras.layers.Conv2D(32, 5, input_shape=(28, 28, 1)))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dropout(0.4))

        # Flatten then pass to dense layer
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.4))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def get_model(self):
        return self.model

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')