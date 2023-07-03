import tensorflow as tf


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
		real_images = batch
		fake_images = self.generator(tf.random.normal((128, 256, 256, 1)), training=False)

		# Train the discriminator
		with tf.GradientTape() as d_tape:
			# Pass the real and fake images to the discriminator model
			yhat_fake = self.discriminator(fake_images, training=True)
			yhat_real = self.discriminator(real_images, training=True)
			yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

			# Create labels for real and fakes images
			y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

			# Add some noise to the TRUE outputs
			noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
			noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
			y_realfake += tf.concat([noise_real, noise_fake], axis=0)

			# Calculate loss - BINARYCROSS
			total_d_loss = self.d_loss(y_realfake, yhat_realfake)

		# Apply backpropagation - nn learn
		dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
		self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

		# Train the generator
		with tf.GradientTape() as g_tape:
			# Generate some new images
			gen_images = self.generator(tf.random.normal((128, 256, 256, 1)), training=True)

			# Create the predicted labels
			predicted_labels = self.discriminator(gen_images, training=False)

			# Calculate loss - trick to training to fake out the discriminator
			total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

		# Apply backprop
		ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
		self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

		return {"d_loss": total_d_loss, "g_loss": total_g_loss}
