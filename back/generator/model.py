from keras import layers
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


SIZE = 256
NUM_IMAGES = 4


def down(filters, kernel_size, apply_batch_normalization=True):
    down_sample = tf.keras.models.Sequential()
    down_sample.add(layers.Conv2D(filters, kernel_size, padding='same', strides=2))
    if apply_batch_normalization:
        down_sample.add(layers.BatchNormalization())
    down_sample.add(layers.LeakyReLU())
    return down_sample


def up(filters, kernel_size, dropout=False):
    up_sample = tf.keras.models.Sequential()
    up_sample.add(layers.Conv2DTranspose(filters, kernel_size, padding='same', strides=2))
    if dropout:
        up_sample.dropout(0.25)
    up_sample.add(layers.LeakyReLU())
    return up_sample


def model():
    inputs = layers.Input(shape=[SIZE, SIZE, 3])
    print(inputs)

    # down sampling
    d1 = down(128, (3, 3), False)(inputs)
    d2 = down(128, (3, 3), False)(d1)
    d3 = down(256, (3, 3), True)(d2)
    d4 = down(512, (3, 3), True)(d3)
    d5 = down(512, (3, 3), True)(d4)

    # up sampling
    u1 = up(512, (3, 3), False)(d5)
    u1 = layers.concatenate([u1, d4])
    u2 = up(256, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d3])
    u3 = up(128, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, d2])
    u4 = up(128, (3, 3), False)(u3)
    u4 = layers.concatenate([u4, d1])
    u5 = up(3, (3, 3), False)(u4)
    u5 = layers.concatenate([u5, inputs])
    output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)


def show_generated_images(generator):
    # Generate image with generator
    generated_images = generator.predict(np.random.randn(NUM_IMAGES, SIZE, SIZE, 3))
    fig, axes = plt.subplots(1, NUM_IMAGES, figsize=(10, 4))

    # Show all images
    for i in range(NUM_IMAGES):
        # Converting the generated image into a format acceptable for display in the "PIL" object
        generated_image_pil = Image.fromarray(np.uint8(generated_images[i] * 255))
        axes[i].imshow(generated_image_pil)
        axes[i].axis('off')

    plt.show()


# generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['acc'])

# generator.fit(train_low_image, train_high_image, epochs=20, batch_size=1, validation_data=(validation_low_image,validation_high_image))
