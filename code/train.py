'''
Code for Rapid Treatment Planning for Low-dose-rate Prostate Brachytherapy with TP-GAN

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia, Vancouver, Canada
'''
import os
import sys
import time
import numpy as np
from keras.utils import generic_utils
from tensorflow.keras.optimizers import Adam, SGD
from skimage.transform import resize
from keras.models import load_model
import keras
import pandas as pd
import datetime
import models
import metrics
from resnet3D import Resnet3DBuilderDiscriminator, Resnet3DBuilderGenerator
import data_utils
import plot_functions

os.environ["KERAS_BACKEND"] = "tensorflow"
# Import the backend
import keras.backend as K
image_data_format = "channels_last"
K.set_image_data_format(image_data_format)


# Initialize the parameters
model_name = "TP-GAN"
use_all_plans = 1 #There are multiple possible plans per case, set to 1 to use all for training

# Dataset name
PTVfilename = "PTV.npy"
CTVfilename = "CTV.npy"
seedsfilename = "Seed_plans.txt"
load_path = "../data/"

batch_size = 2
total_epoch = 1000
lr = 1E-5  # learning rate
B1 = 0.5  # beta 1 momentum parameter
B2 = 0.99  # beta 2 momentum parameter
operation_point = 0.40 # threshold of output, anything above it is set to one
save_model_interval = 50 # save model after given epochs
save_plot_interval = 2 # save plots and model after given epochs

# Logging directory
logging_dir = '../'
data_utils.log_folder(model_name, logging_dir)

# Load and rescale data
Y_train, X_train, Y_val, X_val = data_utils.load_data(load_path, PTVfilename, CTVfilename, seedsfilename, use_all_plans, save_all=0, shuffle=True)

print('\nModel name: ' + model_name)
print('\nTraining samples: ' + str(len(X_train)))
print('Validation samples: ' + str(len(X_val)))

print('\nTraining for PTV + CTV + Needles = Seeds\n')

###
# Augmentation step (setting left and right hand side of data as two different samples
###
half_dimen = 32
X_train = np.concatenate((X_train[:, :, 0:half_dimen, :, :], np.flip(X_train[:, :, half_dimen:, :, :], 2)), axis=0)
Y_train = (Y_train[:, :, 0:6, :, :])
Y_train = np.concatenate((Y_train, Y_train), axis=0)

# No augmentation for validation
X_val = X_val[:, :, 0:half_dimen, :, :]
Y_val = (Y_val[:, :, 0:6, :, :])

batch_size_per_epoch = np.floor(len(X_train) / batch_size)

print("Number of training samples: %s" % len(X_train))
print("Number of batches: %s" % batch_size)
print("Number of batches per epoch: %s\n" % int(batch_size_per_epoch))

epoch_size = batch_size_per_epoch * batch_size

input_img_dim = X_train.shape[-4:]

print("X train Max: " + str(np.max(X_train)) + ", Min: " + str(np.min(X_train)))
print("X val Max: " + str(np.max(X_val)) + ", Min: " + str(np.min(X_val)))
print("Y train Max: " + str(np.max(Y_train)) + ", Min: " + str(np.min(Y_train)))
print("Y val Max: " + str(np.max(Y_val)) + ", Min: " + str(np.min(Y_val)))

print('\nShape of X_train: ' + str(np.shape(X_train)))
print('Shape of Y_train: ' + str(np.shape(Y_train)))
print('\nShape of X_val: ' + str(np.shape(X_val)))
print('Shape of Y_val: ' + str(np.shape(Y_val)))

# Create optimizers
tpgan_optimizer = Adam(lr=lr, beta_1=B1, beta_2=B2, epsilon=1e-08)
discri_optimizer = Adam(lr=lr, beta_1=B1, beta_2=B2, epsilon=1e-08)

# Generator/Encoder network
generator_model = Resnet3DBuilderGenerator.build_generator((64, 32, 14, 3), 2)  # Based on input dimension of PTV and CTV
generator_model.summary()

# Load discriminator network
discriminator_model = Resnet3DBuilderDiscriminator.build_discriminator((64, 32, 14, 4), 2)
discriminator_model.summary()
discriminator_model.trainable = False

# TP-GAN model
TPGAN_model = models.TPGAN(generator_model, discriminator_model, input_img_dim)

# Loss function
adversarial_loss = 'binary_crossentropy'
total_loss = [metrics.l1_and_adj_seed_loss(0.5), adversarial_loss]
loss_weights = [2/3, 1/3]

TPGAN_model.compile(loss=total_loss, loss_weights=loss_weights, optimizer=tpgan_optimizer)
discriminator_model.trainable = True
discriminator_model.compile(loss=adversarial_loss, optimizer=discri_optimizer)


max_auc = 0
max_dice = 0

# Setting the metrics to empty lists
train_operating_point, train_aucs, train_accuracy, train_sensitivity, train_specificity, train_dice, train_needle_diff, train_needle_prohibited, train_seed_diff, train_seed_adj = ([] for _ in range(10))
val_operating_point, val_aucs, val_accuracy, val_sensitivity, val_specificity, val_dice, val_needle_diff, val_needle_prohibited, val_seed_diff, val_seed_adj = ([] for _ in range(10))
epochs_count, disc_loss_all, gen_loss_total, gen_loss_l1_seed_adj, gen_loss_discri = ([] for _ in range(5))

# Start training
print(model_name)
print("\nStart training")

record_df = pd.DataFrame(
	columns=['epoch', 'Train OP', 'Val OP', 'Train AUC', 'Val AUC',
	         'Train Y Dice', 'Val Y Dice', 'Train Sensi', 'Val Sensi', 'Train Speci', 'Val Speci',
	         'Train Needles Diff', 'Val Needles Diff', 'Train Adjacent Needles', 'Val Adjacent Needles',
	         'Train Seeds Diff', 'Val Seeds Diff', 'Train Adjacent Seeds', 'Val Adjacent Seeds',
	         'D Loss', 'G Total', 'G L1+Seed Adj','G Discri'])


for e in range(total_epoch):

	# Initialize progbar and batch counter
	progbar = generic_utils.Progbar(epoch_size)
	batch_counter = 1
	start = time.time()

	for batch_i, (Y_train_batch, X_train_batch, indx) in enumerate(data_utils.generate_batch(Y_train, X_train, batch_size)):

		Y_TPGAN_gt = np.zeros((X_train_batch.shape[0], 2), dtype=np.uint8)
		Y_TPGAN_gt[:, 1] = 1 # GT of discriminator, we want the discriminator to be fooled

		# Freeze the Discriminator
		discriminator_model.trainable = False

		# Update Generator
		gen_loss = TPGAN_model.train_on_batch(X_train_batch, [Y_train_batch, Y_TPGAN_gt])

		# Unfreeze the Discriminator
		discriminator_model.trainable = True

		# Create a batch to feed the discriminator model
		X_disc, y_disc = data_utils.get_disc_batch(Y_train_batch, X_train_batch, generator_model, batch_counter)

		# Update the discriminator
		disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

		progbar.add(batch_size, values=[("D logloss", disc_loss),
										("G Tot", gen_loss[0]),
										("G L1+Seed Adj", gen_loss[1]),
										("G Discri", gen_loss[2])
										])

		batch_counter += 1

	disc_loss_all.append(disc_loss)
	gen_loss_total.append(gen_loss[0])
	gen_loss_l1_seed_adj.append(gen_loss[1])
	gen_loss_discri.append((gen_loss[2]))

	print("\nModel name: " + str(model_name))
	print("")
	print('Epoch %s/%s, Time: %s' % (e + 1, total_epoch, time.time() - start))

	# Predicting results using trained model for current epoch
	y_pred_train = generator_model.predict(X_train, verbose=0)
	y_pred_val = generator_model.predict(X_val, verbose=0)

	# Converting GT to binary
	gt_thresholded_train = Y_train >= 0.6
	gt_thresholded_val = Y_val >= 0.6

	# Calculating all evaluation metrics for Training set
	_, roc_auc_train, accuracy, specificity, sensitivity, dice = data_utils.use_operating_points(operation_point, gt_thresholded_train, y_pred_train)
	needle_diff = data_utils.needle_difference_nomean(np.sum(gt_thresholded_train, axis=3) >= 1, np.sum(y_pred_train >= operation_point, axis=3) >= 1, operation_point)
	seed_diff = data_utils.seed_difference_nomean(gt_thresholded_train, y_pred_train, operation_point)

	prohibited_needles = metrics.find_prohibited_needles(operation_point, (np.sum(y_pred_train >= operation_point, axis=3).astype('float32')))
	adjacent_seeds = metrics.find_adjacent_seeds(operation_point, y_pred_train.astype('float32'))


	print('\n\n')
	print(model_name)
	print(
		"Training Operating Point:{:.4f}, AUC :{:.4f} ({:.4f}), Dice: {"
		":.4f} ({:.4f}), Needle Diff: {:.2f}  ({:.2f}), Seed Diff: {:.2f} ({:.2f}), Prohib Needles: {:.2f} ({:.2f}), Seed Adj: {:.2f} ({:.2f}), Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f},".format(
			operation_point, np.mean(roc_auc_train), np.std(roc_auc_train), np.mean(dice), np.std(dice),
			np.mean(needle_diff), np.std(needle_diff), np.mean(seed_diff), np.std(seed_diff),
			np.sum(prohibited_needles), np.mean(prohibited_needles), np.sum(adjacent_seeds),
			np.mean(adjacent_seeds), accuracy, sensitivity, specificity))

	# Can be used later for drawing plots
	train_operating_point.append(operation_point)
	train_aucs.append(np.mean(roc_auc_train))
	train_accuracy.append(accuracy)
	train_sensitivity.append(sensitivity)
	train_specificity.append(specificity)
	train_dice.append(np.mean(dice))
	train_needle_diff.append(np.mean(needle_diff))
	train_needle_prohibited.append(np.sum(prohibited_needles))
	train_seed_diff.append(np.mean(seed_diff))
	train_seed_adj.append(np.sum(adjacent_seeds))

	#Plotting training and validation results
	# Save images for visualization
	if e % save_plot_interval == 0:

		# Training
		plot_functions.plot_generated_batch(operation_point, Y_train_batch, X_train_batch, generator_model, "training_epoch_" + str(e), logging_dir, model_name)

		# Validation
		idx = np.random.choice(X_val.shape[0], batch_size, replace=False)
		plot_functions.plot_generated_batch(operation_point, Y_val[idx], X_val[idx], generator_model, "validation_epoch" + str(e), logging_dir, model_name)


	# Calculating all evaluation metrics for Validation set
	_, roc_auc_val, accuracy, specificity, sensitivity, dice = data_utils.use_operating_points(operation_point, gt_thresholded_val, y_pred_val)
	needle_diff = data_utils.needle_difference_nomean(np.sum(gt_thresholded_val, axis=3) >= 1, np.sum(y_pred_val >= operation_point, axis=3) >= operation_point, operation_point)
	seed_diff = data_utils.seed_difference_nomean(gt_thresholded_val, y_pred_val, operation_point)

	prohibited_needles = metrics.find_prohibited_needles(operation_point, (np.sum(y_pred_val >= operation_point, axis=3)).astype('float32'))
	adjacent_seeds = metrics.find_adjacent_seeds(operation_point, y_pred_val.astype('float32'))


	print(
		"Validation Operating Point:{:.4f}, AUC :{:.4f} ({:.4f}), Dice: {:.4f} ({:.4f}), Needle Diff: {:.2f}  ({:.2f}), Seed Diff: {:.2f} ({:.2f}),Needle Adj: {:.2f} ({:.2f}), Seed Adj: {:.2f} ({:.2f}), , Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f},".format(
			operation_point, np.mean(roc_auc_val), np.std(roc_auc_val), np.mean(dice), np.std(dice),
			np.mean(needle_diff), np.std(needle_diff), np.mean(seed_diff), np.std(seed_diff),
			np.sum(prohibited_needles), np.mean(prohibited_needles), np.sum(adjacent_seeds),
			np.mean(adjacent_seeds), accuracy, sensitivity, specificity))

	# Can be used later for drawing plots
	val_operating_point.append(operation_point)
	val_aucs.append(np.mean(roc_auc_val))
	val_accuracy.append(accuracy)
	val_sensitivity.append(sensitivity)
	val_specificity.append(specificity)
	val_dice.append(np.mean(dice))
	val_needle_diff.append(np.mean(needle_diff))
	val_needle_prohibited.append(np.sum(prohibited_needles))
	val_seed_diff.append(np.mean(seed_diff))
	val_seed_adj.append(np.sum(adjacent_seeds))

	epochs_count.append(e)

	# Generating CSV file with all metrics
	new_row = {'epoch': e, 'Train OP': train_operating_point[e], 'Val OP': val_operating_point[e],
	           'Train AUC': train_aucs[e], 'Val AUC': val_aucs[e],
	           'Train Y Dice': train_dice[e], 'Val Y Dice': val_dice[e],
	           'Train Sensi': train_sensitivity[e], 'Val Sensi': val_sensitivity[e],
	           'Train Speci': train_specificity[e], 'Val Speci': val_specificity[e],
	           'Train Needles Diff': train_needle_diff[e], 'Val Needles Diff': val_needle_diff[e],
	           'Train Adjacent Needles': train_needle_prohibited[e], 'Val Adjacent Needles': val_needle_prohibited[e],
	           'Train Seeds Diff': train_seed_diff[e], 'Val Seeds Diff': val_seed_diff[e],
	           'Train Adjacent Seeds': train_seed_adj[e], 'Val Adjacent Seeds': val_seed_adj[e],
	           'D Loss': disc_loss_all[e], 'G Total': gen_loss_total[e], 'G L1+Seed Adj': gen_loss_l1_seed_adj[e],
	           'G Discri': gen_loss_discri[e]}

	record_df = record_df.append(new_row, ignore_index=True)
	record_df.to_csv(os.path.join(logging_dir, 'models/%s/all_metrics.csv' % (model_name)), index=0)

	# Save model weights based on max AUC or DICE
	if (np.mean(roc_auc_val) > max_auc):
		print(('\nValidation AUC improved from %f to %f at epoch %f\nSaving Top AUC weights....\n') % (max_auc, np.mean(roc_auc_val), e))
		max_auc = np.mean(roc_auc_val)

		#Saving weights of network based on highest AUC
		gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_top_auc.h5' % (model_name))
		generator_model.save(gen_weights_path, overwrite=True)

	if (np.mean(dice) > max_dice):
		print(('\nValidation DICE improved from %f to %f at epoch %f\nSaving Top DICE weights....\n') % (max_dice, np.mean(dice), e))
		max_dice = np.mean(dice)
		#Saving weights of network based on highest AUC
		gen_weights_path = os.path.join(logging_dir,
										'models/%s/gen_weights_top_dice.h5' % (model_name))
		generator_model.save(gen_weights_path, overwrite=True)

	# Saves weights and metrics after set intervals
	if e % save_model_interval == 0:
		gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
		generator_model.save(gen_weights_path, overwrite=True)

	# Save all metrics as npz file as well
	np.savez(os.path.join(logging_dir, 'models/%s/training_metrics' % (model_name)), name1=train_operating_point, name2=train_aucs, name3=train_accuracy, name4=train_sensitivity, name5=train_specificity, name6=train_dice, name7=train_needle_diff, name8=train_needle_prohibited, name9=train_seed_diff, name10=train_seed_adj)
	np.savez(os.path.join(logging_dir, 'models/%s/val_metrics' % (model_name)), name1=val_operating_point, name2=val_aucs, name3=val_accuracy, name4=val_sensitivity, name5=val_specificity, name6=val_dice, name7=val_needle_diff, name8=val_needle_prohibited, name9=val_seed_diff, name10=val_seed_adj)
	np.savez(os.path.join(logging_dir, 'models/%s/losses' % (model_name)), name1=disc_loss_all, name2=gen_loss_total, name3=gen_loss_l1_seed_adj, name4 = gen_loss_discri)



	plot_functions.plot_training_history(train_seed_adj, val_seed_adj, train_aucs, val_aucs,
										 train_accuracy, val_accuracy, train_sensitivity, val_sensitivity,
										 train_specificity, val_specificity, train_dice, val_dice,
										 epochs_count, logging_dir, model_name)

	plot_functions.plot_training_losses(disc_loss_all, gen_loss_total, gen_loss_l1_seed_adj, gen_loss_discri, epochs_count, logging_dir, model_name)





