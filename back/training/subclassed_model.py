import tensorflow as tf
import numpy as np
from data_preparation.dependencies_and_data import get_hr_images, get_lr_images
from skimage.transform import resize


def convert_images_to_tensors(image_list):
	tensor_list = []

	for image in image_list:
		image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
		tensor_list.append(image_tensor)

	return tuple(tensor_list)


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

	def train_step(self, batch):
		# Get the data
		low_res_images = get_lr_images()
		low_res_images = low_res_images[:128]
		images = get_hr_images()
		images = images[:128]
		image_shape = images[0].shape
		image_array = np.array(images)
		# image_array2 = np.array(low_res_images)
		reshaped_array = np.expand_dims(image_array[:, :, :, 0], axis=-1)
		# reshaped_array2 = np.expand_dims(image_array2[:, :, :, 0], axis=-1)
		# print(reshaped_array.shape)
		new_shape = (128, 28, 28, 1)

		# Reshape the array to (128, 256, 256)
		reshaped_array = np.squeeze(reshaped_array)
		# reshaped_array2 = np.squeeze(reshaped_array2)

		# Resize the array to (128, 28, 28)
		resized_array = np.zeros((new_shape[0], new_shape[1], new_shape[2]))
		# resized_array2 = np.zeros((new_shape[0], new_shape[1], new_shape[2]))
		for i in range(new_shape[0]):
			resized_array[i] = resize(reshaped_array[i], (new_shape[1], new_shape[2]))
			# resized_array2[i] = resize(reshaped_array2[i], (new_shape[1], new_shape[2]))

		# Add the last dimension back to the array to have shape (128, 28, 28, 1)
		final_array = np.expand_dims(resized_array, axis=-1)
		# final_array2 = np.expand_dims(resized_array2, axis=-1)
		real_images = final_array
		# fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)
		low_res_images = np.array(low_res_images)
		low_res_images = low_res_images.reshape(-1, 128)
		fake_images = self.generator(low_res_images, training=False)
		# Train the discriminator
		with tf.GradientTape() as d_tape:
			# Pass the real and fake images to the discriminator model
			yhat_real = self.discriminator(real_images, training=True)
			yhat_fake = self.discriminator(fake_images, training=True)
			yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

			# Create labels for real and fakes images
			y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

			# Add some noise to the TRUE outputs
			noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
			noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
			y_realfake += tf.concat([noise_real, noise_fake], axis=0)

			# Calculate loss - BINARYCROSS
			total_d_loss = self.d_loss(y_realfake, yhat_realfake)

			del yhat_real
			del yhat_fake
			del noise_real
			del noise_fake
			del yhat_realfake

		# Apply backpropagation - nn learn
		dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
		self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

		# Train the generator
		with tf.GradientTape() as g_tape:
			# Generate some new images
			gen_images = self.generator(low_res_images, training=True)

			# Create the predicted labels
			predicted_labels = self.discriminator(gen_images, training=False)

			# Calculate loss - trick to training to fake out the discriminator
			total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

		# Apply backprop
		ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
		self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

		return {"d_loss": total_d_loss, "g_loss": total_g_loss}
