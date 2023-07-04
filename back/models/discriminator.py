import tensorflow as tf
from tensorflow import keras


class Discriminator:
    def __init__(self):
        self.model = keras.models.Sequential()

        # first conv block
        self.model.add(keras.layers.Conv2D(32, 5, input_shape=(28, 28, 1)))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dropout(0.4))

        # second conv block
        self.model.add(keras.layers.Conv2D(64, 5))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dropout(0.4))
    
        # third
        self.model.add(keras.layers.Conv2D(128, 5))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dropout(0.4))
    
        # fourth
        self.model.add(keras.layers.Conv2D(256, 5))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dropout(0.4))
        

        # flatten then pass to dense layer
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.4))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def get_model(self):
        return self.model

# enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')