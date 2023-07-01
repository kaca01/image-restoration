import numpy as np
from sklearn.utils import shuffle
import random


def scale_images(images):
    scaled_images = []
    for image in images:
        scaled_image = image / 255
        scaled_images.append(scaled_image)
    return scaled_images


def prepare_images(images):
    scaled_images = scale_images(images)

    random.shuffle(scaled_images)

    # batch the images into 128 images per sample
    batch_size = 128
    batched_images = [scaled_images[i:i+batch_size] for i in range(0, len(scaled_images), batch_size)]

    # Reduce the likelihood of bottlenecking
    prefetched_batches = [batched_images[i:i+64] for i in range(0, len(batched_images), 64)]
    return prefetched_batches
