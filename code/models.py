'''
Code for Rapid Treatment Planning for Low-dose-rate Prostate Brachytherapy with TP-GAN

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia, Vancouver, Canada
'''
from keras.models import Model
from keras.layers import Input, Concatenate
import keras.backend as K
from keras.layers import Lambda
import tensorflow as tf

def TPGAN(generator_model, discriminator_model, img_dim):

	gen_input = Input(shape=img_dim, name="TPGAN_Input")
	generated_image = generator_model(gen_input)

	out0 = Lambda(lambda image: tf.image.resize(image[:, :, :, 0, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out1 = Lambda(lambda image: tf.image.resize(image[:, :, :, 1, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out2 = Lambda(lambda image: tf.image.resize(image[:, :, :, 2, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out3 = Lambda(lambda image: tf.image.resize(image[:, :, :, 3, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out4 = Lambda(lambda image: tf.image.resize(image[:, :, :, 4, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out5 = Lambda(lambda image: tf.image.resize(image[:, :, :, 5, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out6 = Lambda(lambda image: tf.image.resize(image[:, :, :, 6, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out7 = Lambda(lambda image: tf.image.resize(image[:, :, :, 7, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out8 = Lambda(lambda image: tf.image.resize(image[:, :, :, 8, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out9 = Lambda(lambda image: tf.image.resize(image[:, :, :, 9, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out10 = Lambda(lambda image: tf.image.resize(image[:, :, :, 10, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out11 = Lambda(lambda image: tf.image.resize(image[:, :, :, 11, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out12 = Lambda(lambda image: tf.image.resize(image[:, :, :, 12, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)
	out13 = Lambda(lambda image: tf.image.resize(image[:, :, :, 13, :], (64, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
	                                           preserve_aspect_ratio=False))(generated_image)


	out = Concatenate(axis=-1)([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13])

	out = Lambda(lambda image: K.expand_dims(image, axis=-1))(out)
	out = Concatenate(axis=-1)([gen_input, out])

	discri_out = discriminator_model(out)

	TPGAN = Model(inputs=[gen_input], outputs=[generated_image, discri_out], name="TPGAN")

	TPGAN.summary()

	return TPGAN



