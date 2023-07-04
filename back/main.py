from data_preparation.dependencies_and_data import preparation, get_hr_images, get_lr_images
from data_preparation.image_preparation import prepare_images
from models.generator import Generator
from models.discriminator import Discriminator
from training.subclassed_model import GAN
from training.monitoring import ModelMonitor
from tensorflow.python.profiler import profiler_v2
import tensorflow as tf
import numpy as np

# preparation()
# # images = high()
# # prefetched_batches = reshape_images(images)
images = get_hr_images()
prefetched_batches = prepare_images(images)
generator = Generator()
gen_model = generator.get_model()

low_res_images = tf.data.Dataset.from_generator(get_lr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
low_res_images = low_res_images.batch(128).prefetch(1)
		
high_res_images = tf.data.Dataset.from_generator(get_hr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
high_res_images = high_res_images.batch(128).prefetch(1)
# gen_model.summary()
# # generator.show_generated_images(gen_model)

discriminator = Discriminator()
disc_model = discriminator.get_model()
# disc_model.summary()

# # this image should be from generator
# # print(disc_model.predict(image))

# # Create instance of subclassed model
# profiler_v2.start(logdir='profiler_logs/')
gan = GAN(gen_model, disc_model)
# # Compile the model
gan.compile()

low_res_images = get_lr_images()
low_res_images = tf.data.Dataset.from_generator(get_lr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
low_res_images = low_res_images.batch(32).prefetch(1)

high_res_images = get_hr_images()
high_res_images = tf.data.Dataset.from_generator(get_hr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
high_res_images = high_res_images.batch(32).prefetch(1)

zip_batches = zip(low_res_images, high_res_images)

# # Recommend 2000 epochs
hist = gan.fit(zip_batches, epochs=20, batch_size=32, callbacks=[ModelMonitor()])
profiler_v2.stop()

# Set the number of epochs and batch size
# def preprocess_images(images):
#     images_array = np.array(images)

#     # Normalize the pixel values to the range [-1, 1]
#     normalized_images = (images_array.astype(np.float32) - 127.5) / 127.5

#     # Reshape the images to the desired shape
#     reshaped_images = tf.reshape(normalized_images, (-1, 28, 28, 1))

#     return reshaped_images

# epochs = 10
# batch_size = 64

# # Create the GAN model
# # gan = GAN()

# # Compile the GAN model
# gan.compile()

# # Get the low-resolution and high-resolution images
# lr_images = get_lr_images()
# hr_images = get_hr_images()

# # print("Length: " + str(len(lr_images) + str(hr_images)))

# # Preprocess the images
# preprocessed_lr_images = preprocess_images(lr_images)
# preprocessed_hr_images = preprocess_images(hr_images)

# # Create TensorFlow Dataset objects for training
# train_dataset = tf.data.Dataset.from_tensor_slices((preprocessed_lr_images, preprocessed_hr_images)).batch(batch_size)

# # Define the checkpoint callback to save the model weights
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('gan_checkpoint.h5', save_weights_only=True, save_best_only=True)

# Train the GAN model
# gan.fit(x=train_dataset, y=None, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint_callback], verbose=1, shuffle=True, initial_epoch=0)

