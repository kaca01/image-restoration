from data_preparation.dependencies_and_data import preparation, get_hr_images
from data_preparation.image_preparation import prepare_images
from models.generator import Generator
from models.discriminator import Discriminator

preparation()

images = get_hr_images()

prepare_images(images)

generator = Generator()
gen_model = generator.model()
# gen_model.summary()
generator.show_generated_images(gen_model)

discriminator = Discriminator()
disc_model = discriminator.get_model()
disc_model.summary()

# this image should be from generator
# print(disc_model.predict(image))
