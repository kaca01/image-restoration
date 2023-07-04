import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2


def preparation():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        # print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    os.environ['KAGGLE_CONFIG_DIR'] = 'C:/Users/User/.kaggle'


def image_generator(dataset_dir, limit=200):
    count = 0
    for filename in os.listdir(dataset_dir):
        if count >= limit:
            break
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(dataset_dir, filename)
            image = plt.imread(image_path)
            yield image
            count += 1


def get_lr_images():
    lr_dataset_dir = 'datasets/dataset/train/low_res'
    return image_generator(lr_dataset_dir)


def get_hr_images():
    hr_dataset_dir = 'datasets/dataset/train/high_res'
    return image_generator(hr_dataset_dir)

# Usage example:
# lr_images = get_lr_images()
# hr_images = get_hr_images()

# Convert generator to tf.data.Dataset
# lr_dataset = tf.data.Dataset.from_generator(get_lr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
# hr_dataset = tf.data.Dataset.from_generator(get_hr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))

# def high():
#     high_img = []
#     path = 'datasets/dataset/Raw Data/high_res'
#     files = os.listdir(path)
#     files = sorted(files, key=lambda x: int(x.split('.')[0]))  # Sort files numerically
#     for i in tqdm(files):
#         if i == '855.jpg':
#             break
#         else:
#             img = cv2.imread(path + '/' + i, 1)
#             # open cv reads images in BGR format so we have to convert it to RGB
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             # resizing image
#             img = cv2.resize(img, (256, 256))
#             img = img.astype('float32') / 255.0
#             high_img.append(tf.keras.preprocessing.image.array_to_img(img))

#     # return np.array(high_img)
#     # return high_img
