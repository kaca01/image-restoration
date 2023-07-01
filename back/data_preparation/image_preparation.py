import numpy as np
from sklearn.utils import shuffle


def scale_images(data):
    scaled_images = []
    for image in data:
        scaled_image = image / 255.0
        scaled_images.append(scaled_image)
    return np.array(scaled_images)


def prepare_images(images):
    # image_array = np.array(images)
    #
    # # Scale the images (preprocessing step)
    # scaled_images = scale_images(image_array)
    #
    # # Shuffle the images
    # shuffled_images = shuffle(scaled_images)  # if needed
    #
    # # Batch the images
    # batch_size = 128
    # num_samples = shuffled_images.shape[0]
    # num_batches = num_samples // batch_size
    #
    # batches = np.array_split(shuffled_images, num_batches)
    #
    # # Prefetch the batches
    # prefetched_batches = [batch for batch in batches]
    # print(prefetched_batches)
    # return prefetched_batches

    image_array = np.array(images)

    # Scale the images (preprocessing step)
    scaled_images = scale_images(image_array)  # replace 'scale_images' with your own scaling function

    # Shuffle the images
    shuffled_images = shuffle(scaled_images)  # if needed

    # Calculate the number of samples and batches
    num_samples = len(shuffled_images)
    batch_size = 128
    num_batches = num_samples // batch_size

    # Check if the number of batches is greater than zero
    if num_batches > 0:
        batches = np.array_split(shuffled_images, num_batches)

        # Prefetch the batches
        prefetched_batches = [batch for batch in batches]

        # Now you can iterate over the prefetched batches and perform further operations
        for batch in prefetched_batches:
            pass
        return prefetched_batches
    # Perform operations on the batch
    # ...
    else:
        print("Number of samples is less than the batch size.")
