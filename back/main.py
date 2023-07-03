from data_preparation.dependencies_and_data import preparation, get_hr_images, high
from data_preparation.image_preparation import prepare_images, reshape_images
from models.generator import Generator
from models.discriminator import Discriminator
from training.subclassed_model import GAN
from training.monitoring import ModelMonitor
from tensorflow.python.profiler import profiler_v2

preparation()
# images = high()
# prefetched_batches = reshape_images(images)
images = get_hr_images()
prefetched_batches = prepare_images(images)
generator = Generator()
gen_model = generator.get_model()
gen_model.summary()
# generator.show_generated_images(gen_model)

discriminator = Discriminator()
disc_model = discriminator.get_model()
disc_model.summary()

# this image should be from generator
# print(disc_model.predict(image))

# Create instance of subclassed model
profiler_v2.start(logdir='profiler_logs/')
gan = GAN(gen_model, disc_model)
# Compile the model
gan.compile()

# Recommend 2000 epochs
hist = gan.fit(prefetched_batches, epochs=20, batch_size=2, callbacks=[ModelMonitor()])
profiler_v2.stop()
