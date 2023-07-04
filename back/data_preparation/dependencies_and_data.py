import tensorflow as tf
import os
from matplotlib import pyplot as plt


def preparation():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.environ['KAGGLE_CONFIG_DIR'] = 'C:/Users/User/.kaggle'


def image_generator(dataset_dir):
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(dataset_dir, filename)
            image = plt.imread(image_path)
            yield image


def get_lr_images():
    lr_dataset_dir = 'datasets/dataset/train/low_res'
    return image_generator(lr_dataset_dir)


def get_hr_images():
    hr_dataset_dir = 'datasets/dataset/train/high_res'
    return image_generator(hr_dataset_dir)