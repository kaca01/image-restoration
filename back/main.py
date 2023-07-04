from data_preparation.dependencies_and_data import preparation, get_hr_images, get_lr_images
from data_preparation.image_preparation import prepare_images
from models.generator import Generator
from models.discriminator import Discriminator
from training.subclassed_model import GAN
from training.monitoring import ModelMonitor
import tensorflow as tf


preparation()

low_res_images = prepare_images(True)
		
high_res_images = prepare_images(False)

# creating models
generator = Generator()
gen_model = generator.get_model()
gen_model.summary()
# generator.show_generated_images(gen_model)

discriminator = Discriminator()
disc_model = discriminator.get_model()
disc_model.summary()

# gan
gan = GAN(gen_model, disc_model)
gan.compile()

zip_batches = zip(low_res_images, high_res_images)

hist = gan.fit(zip_batches, epochs=2000, batch_size=128, callbacks=[ModelMonitor()])
