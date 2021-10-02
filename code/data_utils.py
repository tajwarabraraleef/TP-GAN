'''
Code for Rapid Treatment Planning for Low-dose-rate Prostate Brachytherapy with TP-GAN

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca
Robotics and Control Laboratory, University of British Columbia, Vancouver, Canada
'''

import numpy as np
import pickle
import os
from sklearn.metrics import roc_curve, auc
from scipy import ndimage
from skimage.transform import resize
import plot_functions

def log_folder(model_name, log_dir):
	main_model_dir = os.path.join(log_dir, 'models')
	model_dir = os.path.join(log_dir, 'models', model_name)
	fig_dir = os.path.join(log_dir, 'models', model_name, 'visualization')
	create_dir([main_model_dir, model_dir, fig_dir])

def create_dir(dirs):
	"""
	Create directory
	args: dirs (str or list) create all dirs in 'dirs'
	"""

	if isinstance(dirs, (list, tuple)):
		for d in dirs:
			if not os.path.exists(os.path.expanduser(d)):
				os.makedirs(d)
	elif isinstance(dirs, str):
		if not os.path.exists(os.path.expanduser(dirs)):
			os.makedirs(dirs)


def load_data(load_path, PTVfilename, CTVfilename, seedsfilename, use_all_plans, save_all=1, shuffle=True):

	"""
	:param load_path: Load path
	:param PTVfilename: PTV filename
	:param CTVfilename: CTV filename
	:param seedsfilename: Seed plan filename
	:param use_all_plans: Set as 1 to use all plans for training
	:param save_all: save loaded data for visualization
	:param shuffle: Set as True to shuffle data after loading
	:return: Returns training and validation data
	"""

	print('\nLoading Data.....')
	Xptv = np.load(load_path + PTVfilename)
	Xctv = np.load(load_path + CTVfilename)
	with open(load_path + seedsfilename, "rb") as fp:  # Can have multiple plans for every given case
		Yseeds = pickle.load(fp)

	indexes = np.arange(len(Xptv))
	if shuffle:
		np.random.seed(10)
		print('Shuffling Dataset...')
		np.random.shuffle(indexes)
		Xptv = Xptv[indexes]
		Xctv = Xctv[indexes]
		np.random.seed(10)
		np.random.shuffle(Yseeds)

	# Splitting training data by 80% for training and rest for validation
	split = int(np.round(np.shape(Xptv)[0] * 0.6)) #Now set to 60% because the sample data provided is very small
	Xptv_train_main = Xptv[0:split]
	Xctv_train_main = Xctv[0:split]
	Y_train_main = Yseeds[0:split]

	# Unwrapping multiple plans for training set
	X_train = []
	Y_train = []

	for i in range(np.shape(Xptv_train_main)[0]):
		Ytemp = np.asarray(Y_train_main[i])
		num_plans = np.shape(Ytemp)[4]
		if use_all_plans != 1:
			num_plans = 1  # If we only want to use main plan for training
		for j in range(num_plans):
				X_train.append(np.concatenate((np.expand_dims(Xptv_train_main[i], axis=-1), np.expand_dims(Xctv_train_main[i], axis=-1)), axis=-1)) # concatenating PTV and CTV
				Y_train.append(Ytemp[0, :, :, :, j])

	X_train = (np.asarray(X_train)).astype(np.float32)
	Y_train = (np.asarray(Y_train)).astype(np.float32)
	Y_train = np.expand_dims(Y_train, axis=-1)

	# Splitting Validation data
	Xptv_val_main = Xptv[split:]
	Xctv_val_main = Xctv[split:]
	Y_val_main = Yseeds[split:]

	X_val = []
	Y_val = []
	for i in range(np.shape(Xptv_val_main)[0]):
		Ytemp = np.asarray(Y_val_main[i])
		num_plans = 1 # Using only main plans for validation
		for j in range(num_plans):
			X_val.append(np.concatenate((np.expand_dims(Xptv_val_main[i], axis=-1), np.expand_dims(Xctv_val_main[i], axis=-1)), axis=-1))
			Y_val.append(Ytemp[0, :, :, :, j])

	X_val = (np.asarray(X_val)).astype(np.float32)
	Y_val = (np.asarray(Y_val)).astype(np.float32)
	Y_val = np.expand_dims(Y_val, axis=-1)

	# Finding needle plans from seed plans
	Y_train_needles = np.sum(Y_train > 0.6, axis=-2)
	Y_val_needles = np.sum(Y_val > 0.6, axis=-2)

	# Here using the actual needle plans but for validation, its better to use predicted needle plans from [1].
	# Because we are considering this as an end-to-end technique and the needle plans that will be used during
	# inference will be the predicted ones from [1]

	# [1] Aleef, T.A., Spadinger, I.T., Peacock, M.D., Salcudean, S.E.,
	# Mahdavi, S.S.: Centre-specific autonomous treatment plans for prostate
	# brachytherapy using CGANs. Int. J. Comput. Assist. Radiol. Surg., 1–10 (2021)

	# For binarizing needles and performing distance transform
	dt_weight_factor = 3  # the weight determines how much weight to give to positive pixels
	Y_train_needles = weighted_distance_trans(Y_train_needles >= 1, dt_weight_factor=dt_weight_factor)
	Y_val_needles = weighted_distance_trans(Y_val_needles >= 1, dt_weight_factor=dt_weight_factor)

	# Resize to match the input size of PTV and CTV
	Y_train_needles = (resize(Y_train_needles, (len(Y_train_needles), np.shape(X_train)[1], np.shape(X_train)[2], 1), order=0, preserve_range=True))
	Y_val_needles = (resize(Y_val_needles, (len(Y_val_needles), np.shape(X_val)[1], np.shape(X_val)[2], 1), order=0, preserve_range=True))

	# Tile to match dimension in last axis
	Y_train_needles = np.tile(Y_train_needles, (1, 1, 1, np.shape(X_train)[3]))
	Y_train_needles = np.expand_dims(Y_train_needles, axis=-1)
	Y_val_needles = np.tile(Y_val_needles, (1, 1, 1, np.shape(X_val)[3]))
	Y_val_needles = np.expand_dims(Y_val_needles, axis=-1)

	# For binarizing seeds and performing distance transform on all data
	Y_train = weighted_distance_trans(Y_train > 0.6, dt_weight_factor=dt_weight_factor)
	Y_val = weighted_distance_trans(Y_val > 0.6, dt_weight_factor=dt_weight_factor)

	X_train[:, :, :, :, 0] = weighted_distance_trans(X_train[:, :, :, :, 0] >= 0.6, dt_weight_factor=dt_weight_factor)
	X_val[:, :, :, :, 0] = weighted_distance_trans(X_val[:, :, :, :, 0] >= 0.6, dt_weight_factor=dt_weight_factor)

	X_train[:, :, :, :, 1] = weighted_distance_trans(X_train[:, :, :, :, 1] >= 0.6, dt_weight_factor=dt_weight_factor)
	X_val[:, :, :, :, 1] = weighted_distance_trans(X_val[:, :, :, :, 1] >= 0.6, dt_weight_factor=dt_weight_factor)

	X_train = np.concatenate((X_train, Y_train_needles), axis=-1)
	X_val = np.concatenate((X_val, Y_val_needles), axis=-1)

	if save_all:
		plot_functions.plot_all_data(X_train, Y_train, load_path, 'Training', 0)
		plot_functions.plot_all_data(X_val, Y_val, load_path, 'Validation', 0)

	return Y_train, X_train, Y_val, X_val

def weighted_distance_trans(Y, dt_weight_factor=3):
	"""
	This function calculates the weighted distance transform of a binary matrix
	:param Y: 4D Binary matrix
	:param dt_weight_factor: the weight factor on positive pixels
	:return: Weighted distance transformed matrix
	"""
	wdt = []

	for i in range(len(Y)):
		temp = ndimage.distance_transform_edt(np.logical_not(Y[i]))
		temp = abs(temp - np.max(temp))
		temp = temp / np.max(temp)
		temp[temp == 1] = dt_weight_factor
		wdt.append(temp/dt_weight_factor)

	return np.asarray(wdt)


# Generate batch of data
def generate_batch(Y, X, batch_size):

	number_of_batches = int(len(X)/batch_size)
	total_sample = number_of_batches*batch_size
	idx = np.random.choice(X.shape[0], total_sample, replace=False)

	for i in range(number_of_batches - 1):
		idx_ = idx[i * batch_size:(i + 1) * batch_size]

		yield Y[idx_], X[idx_] ,idx_


def get_disc_batch(Y_train_batch, X_train_batch, generator_model, batch_counter):
	# Create X_disc: alternatively only generated or real images
	if batch_counter % 2 == 0:
		# Produce an output
		X_disc = generator_model.predict(X_train_batch)
		X_disc = (resize(X_disc, (len(X_disc), np.shape(X_train_batch)[1], np.shape(X_train_batch)[2], np.shape(X_train_batch)[3], 1), order=0, preserve_range=True))
		X_disc = np.concatenate((X_train_batch, X_disc), axis=-1)
		y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
		y_disc[:, 0] = 1

	else:
		X_disc = Y_train_batch
		X_disc = (resize(X_disc, (len(X_disc), np.shape(X_train_batch)[1], np.shape(X_train_batch)[2], np.shape(X_train_batch)[3], 1), order=0, preserve_range=True))
		X_disc = np.concatenate((X_train_batch, X_disc), axis=-1)
		y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
		y_disc[:, 1] = 1

	return X_disc, y_disc


def use_operating_points(operation_point, y_gt, y_pr):
	# Calculate evaluation metrics based on operating point (threshold)

	# Find Accuracy, Specificity and Sensitivity
	auc_ = []
	accuracy = []
	specificity = []
	sensitivity = []
	dice = []
	for i in range(np.shape(y_gt)[0]):
		fpr, tpr, _ = roc_curve(y_gt[i, :, :, :].flatten(), y_pr[i, :, :, :].flatten())
		auc_.append(auc(fpr, tpr))

		y_temp = y_pr[i, :, :, :].flatten()
		y_temp = y_temp >= operation_point
		TP, FP, TN, FN = perf_measure(y_gt[i, :, :, :].flatten(), y_temp)
		accuracy.append((TP + TN) / (TP + FP + TN + FN))
		fpr_tempp = (FP / (FP + TN))
		tpr_tempp = (TP / (TP + FN))

		specificity.append((1 - fpr_tempp))
		sensitivity.append(tpr_tempp)
		dice.append((2 * TP) / ((2 * TP) + FP + FN))

	return operation_point, auc_, np.mean(accuracy, axis=0), np.mean(specificity, axis=0), np.mean(
		sensitivity, axis=0), dice

def perf_measure(y_actual, y_hat):
	TP = np.logical_and(y_actual, y_hat)
	FP = np.logical_and(y_hat, abs(y_actual-1))
	TN = np.logical_and(abs(y_hat-1), abs(y_actual-1))
	FN = np.logical_and(y_actual, abs(y_hat-1))

	return(np.sum(TP), np.sum(FP), np.sum(TN), np.sum(FN))

def needle_difference_nomean(y_gt, y_pr, operation_point):
	return (abs(np.sum(y_gt > 0.6, axis=(1, 2, 3)) - np.sum(y_pr > operation_point, axis=(1, 2, 3))))

def seed_difference_nomean(y_gt, y_pr, operation_point):
	return (abs(np.sum(y_gt > 0.6, axis = (1,2,3,4)) - np.sum(y_pr > operation_point, axis = (1,2,3,4))))

def seed_needle_count(seeds_pred, seeds_gt, operating_point = 0.5):

	seeds_pred = np.squeeze(np.float32(seeds_pred >= operating_point))
	seeds_gt = np.squeeze(np.float32(seeds_gt >= 0.6))

	# number of seeds
	seeds_pred_no = np.sum(seeds_pred, axis=(0, 1, 2))
	seeds_gt_no = np.sum(seeds_gt, axis=(0, 1, 2))

	# number of needles
	needles_pred_no = np.sum(seeds_pred, axis=-1)
	needles_gt_no = np.sum(seeds_gt, axis=-1)

	needles_pred_no = needles_pred_no >= 1
	needles_gt_no = needles_gt_no >= 1

	needles_pred_no = np.sum(needles_pred_no, axis=(0, 1))
	needles_gt_no = np.sum(needles_gt_no, axis=(0, 1))

	return seeds_pred_no, seeds_gt_no, needles_pred_no, needles_gt_no











