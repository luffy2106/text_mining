from __future__ import division
import numpy as np

import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

def boost(num_learners, weak_predictions, T, y_train):
	"""
	Parameters:

	weak_predictions: an array whose the i-th row contains predictions of the 
	i-th weak learner on training data
	y_train: an array of true labels. Possible labels are -1 and 1
	"""
	n = weak_predictions.shape[1]
	weight = np.array([1/n] * n)
	learner_contribution = []

	# compute wrong predictions
	index_wrong_predictions = []
	for i in range(num_learners):
		index_wrong_predictions.append(
			np.array(range(n))[weak_predictions[i, :] != y_train])
	# print index_wrong_predictions

	for iteration in range(T):
		# find weak learner that minimizes error
		errors = [np.sum(weight[index_wrong_predictions[i]]) 
				  for i in range(num_learners)]
		# print errors
		argmin = np.argmin(errors)
		# print argmin
		alpha = 0.5*np.log((1-errors[argmin])/errors[argmin])
		learner_contribution.append((argmin, alpha))

		# update weight
		new_weight = np.multiply(weight, 
					 np.exp(-alpha * np.multiply(y_train, 
					 		 				    weak_predictions[argmin, :])))
		weight = new_weight/np.sum(new_weight)

	return learner_contribution