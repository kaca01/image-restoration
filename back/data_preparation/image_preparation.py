import tensorflow as tf
from data_preparation.dependencies_and_data import get_hr_images, get_lr_images


def scale_images(images):
    scaled_images = []
    for image in images:
        scaled_image = image / 255
        scaled_images.append(scaled_image)
    return scaled_images


def prepare_images(lr=True):
    if lr:
        images = tf.data.Dataset.from_generator(get_lr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
    else: 
        images = tf.data.Dataset.from_generator(get_hr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))

    images = images.batch(128).prefetch(64)
    images = images.cache()
    images = images.shuffle(60000)
    return images
