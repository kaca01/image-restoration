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

	# @tf.custom_gradient
	# def gradient_checkpointing(x):
	# 	return tf.identity(x), lambda dy: dy

	# @tf.function
	def train_step(self, batch):
		# batches
		low_res_images = get_lr_images()[:32]
		images = get_hr_images()[:32]


		# preprocessing data
		resized_images = np.array([resize(image[:, :, 0], (28, 28)) for image in images])
		real_images = np.expand_dims(resized_images, axis=-1)
		del resized_images
		del images

		# generate fake images
		low_res_images = np.array(low_res_images).reshape(-1, 128)
		fake_images = self.generator(low_res_images, training=False)

		print("discriminator")
		with tf.GradientTape() as d_tape:
			yhat_real = self.discriminator(real_images, training=True)
			yhat_fake = self.discriminator(fake_images, training=True)
			yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
			y_realfake = tf.concat([tf.zeros_like(yhat_real, dtype=tf.float32), tf.ones_like(yhat_fake, dtype=tf.float32)], axis=0)
			noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real), dtype=tf.float32)
			noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake), dtype=tf.float32)
			y_realfake += tf.concat([noise_real, noise_fake], axis=0)
			total_d_loss = self.d_loss(y_realfake, yhat_realfake)
		
		del yhat_real
		del real_images
		del fake_images
		del yhat_fake
		del yhat_realfake
		del y_realfake
		del noise_real
		del noise_fake
		dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
		self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
		del d_tape
		del dgrad

		print("generator")
		with tf.GradientTape() as g_tape:
			gen_images = self.generator(low_res_images, training=True)
			del low_res_images
			predicted_labels = self.discriminator(gen_images, training=False)
			total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)
			del predicted_labels

		ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
		del g_tape
		self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
		del ggrad

		return {"d_loss": total_d_loss, "g_loss": total_g_loss}

