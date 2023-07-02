from data_preparation.dependencies_and_data import preparation, get_lr_images, get_hr_images
from data_preparation.image_preparation import prepare_images
from generator.model import model, show_generated_images

preparation()

images = get_hr_images()

prepare_images(images)

generator = model()
# generator.summary()
show_generated_images(generator)
