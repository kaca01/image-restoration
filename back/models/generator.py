from tensorflow import keras


class Generator:
    def __init__(self):
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Dense(7 * 7 * 128, input_dim=128))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Reshape((7, 7, 128)))

        self.model.add(keras.layers.UpSampling2D())
        self.model.add(keras.layers.Conv2D(128, 5, padding='same'))
        self.model.add(keras.layers.LeakyReLU(0.2))
    
        self.model.add(keras.layers.UpSampling2D())
        self.model.add(keras.layers.Conv2D(128, 5, padding='same'))
        self.model.add(keras.layers.LeakyReLU(0.2))
    
        self.model.add(keras.layers.Conv2D(128, 4, padding='same'))
        self.model.add(keras.layers.LeakyReLU(0.2))
    
        self.model.add(keras.layers.Conv2D(128, 4, padding='same'))
        self.model.add(keras.layers.LeakyReLU(0.2))

        self.model.add(keras.layers.Conv2D(1, 4, padding='same', activation='sigmoid'))

    def get_model(self):
        return self.model