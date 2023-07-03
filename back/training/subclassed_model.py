import tensorflow as tf
import numpy as np
from data_preparation.dependencies_and_data import get_hr_images, get_lr_images
from skimage.transform import resize
import gc


def convert_images_to_tensors(image_list):
	tensor_list = []

	for image in image_list:
		image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
		tensor_list.append(image_tensor)

	return tuple(tensor_list)

def preprocess_images(images, discriminator=True):
    if discriminator:
        reshaped_images = tf.reshape(images, (-1, 28, 28, 1))
    else:
        reshaped_images = tf.reshape(images, (-1, 128))
    return reshaped_images


# subclassed model
class GAN(tf.keras.models.Model):
	def __init__(self, generator, discriminator, *args, **kwargs):
		# pass through args and kwargs to base class
		super().__init__(*args, **kwargs)

		# create attributes for gen and disc
		self.generator = generator
		self.discriminator = discriminator
		# optimizers and losses
		self.g_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
		self.d_opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
		self.g_loss = tf.keras.losses.BinaryCrossentropy()
		self.d_loss = tf.keras.losses.BinaryCrossentropy()

	def compile(self, *args, **kwargs):
		# Compile with base class
		super().compile(*args, **kwargs)


	# @tf.function
	def train_step(self, batch):
		# batches
		low_res_images = get_lr_images()
		low_res_images = tf.data.Dataset.from_generator(get_lr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
		low_res_images = low_res_images.batch(128).prefetch(1)

		high_res_images = get_hr_images()
		high_res_images = tf.data.Dataset.from_generator(get_hr_images, output_signature=tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
		high_res_images = high_res_images.batch(128).prefetch(1)

		# Iterate over the batches
		for low_res_batch, high_res_batch in zip(low_res_images, high_res_images):
			# Preprocess the images
			low_res_images = preprocess_images(low_res_batch, discriminator=False)
			high_res_images = preprocess_images(high_res_batch, discriminator=True)

			# Generate fake images
			fake_images = self.generator(low_res_images, training=False)
			print("discriminator")
			# Train the discriminator
			with tf.GradientTape() as d_tape:
				# Pass the real and fake images to the discriminator model
				yhat_real = self.discriminator(high_res_images, training=True)
				yhat_fake = self.discriminator(fake_images, training=True)
				yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

				# Create labels for real and fake images
				y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

				# Add some noise to the TRUE outputs
				noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
				noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
				y_realfake += tf.cast(tf.concat([noise_real, noise_fake], axis=0), dtype=tf.float16)

				# Calculate discriminator loss
				total_d_loss = self.d_loss(y_realfake, yhat_realfake)

			# Calculate gradients and update discriminator weights
			dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
			self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
			print("generator")
			# Train the generator
			with tf.GradientTape() as g_tape:
				# Generate some new images
				gen_images = self.generator(low_res_images, training=True)

				# Create the predicted labels
				predicted_labels = self.discriminator(gen_images, training=False)

				# Calculate generator loss
				total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

			# Calculate gradients and update generator weights
			ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
			self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

		return {"d_loss": total_d_loss, "g_loss": total_g_loss}

