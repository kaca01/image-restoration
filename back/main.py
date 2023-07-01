from data_preparation.dependencies_and_data import preparation, get_lr_images, get_hr_images
from data_preparation.image_preparation import prepare_images
from models.Discriminator import Discriminator

preparation()

images = get_hr_images()

images = prepare_images(images)

discriminator = Discriminator()
disc_model = discriminator.get_model()
disc_model.summary()

# this image should be from generator
# print(disc_model.predict(image))
