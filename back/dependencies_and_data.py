import tensorflow as tf
import os
from matplotlib import pyplot as plt


def preparation():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    # if you don't have this dir, create it
    # and replace with your location
    # in this directory put your credentials from kaggle
    os.environ['KAGGLE_CONFIG_DIR'] = 'C:/Users/User/.kaggle'


def get_images(dataset_dir):
    images = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(dataset_dir, filename)
            image = plt.imread(image_path)
            images.append(image)

    # display one of the images
    plt.imshow(images[0])
    plt.axis('off')
    plt.show()


def get_lr_images():
    lr_dataset_dir = 'datasets/Data/LR'
    return get_images(lr_dataset_dir)


def get_hr_images():
    hr_dataset_dir = 'datasets/Data/HR'
    return get_images(hr_dataset_dir)
