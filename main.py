import util
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

def apply_subset(x_train, y_train, x_test, y_test,  ml_algorithm, predictors, give_clf = False):
	x_train = x_train[predictors]
	x_test = x_test[predictors]
	
	return(ml_algorithm(x_train, y_train, x_test, y_test, give_clf))

def best_subset(x_train, y_train, x_test, y_test,  ml_algorithm, predictors = None):
	# qda: (['state_lattitude', 'last_funding_at', 'funding_rounds', 'state_importance', 'city_importance', 'last_funding_at_year'], 0.8022300931257897)]
	preds = [ 'state_lattitude']
	if predictors is None:
		predictors = ['first_funding_at', 'first_funding_at_year', \
			'last_funding_at', 'funding_rounds', 'state_importance', 'city_importance', 'last_funding_at_year', 'last_milestone_at', \
			'last_milestone_at_year', 'founded_at_year','founded_at', 'funding_total_usd']
	scores = []
	for p in predictors:
		preds_to_try = []
		preds_to_try.extend(preds)
		preds_to_try.append(p)
		score = apply_subset(x_train, y_train, x_test, y_test, ml_algorithm, preds_to_try)
		print(scores)
		scores.append((preds_to_try, score))
	scores.sort(key = lambda x: x[1])
	return(scores)

def forward_selection(x_train, y_train, x_test, y_test, ml_algorithm, unused_preds = ['funding_total_usd', 'funding_rounds', 'founded_at', 'first_funding_at',\
       'last_funding_at', 'last_milestone_at', 'state_importance',\
       'state_longitude', 'state_lattitude', 'city_importance',\
       'city_longitude', 'city_lattitude', 'last_milestone_at_month',\
       'last_milestone_at_year', 'founded_at_month', 'founded_at_year',\
       'first_funding_at_month', 'first_funding_at_year',\
       'last_funding_at_month', 'last_funding_at_year'], used_preds = [], scores = []):
	if len(unused_preds) == 0:
		return(scores)
	else:
		new_scores = []
		for p in unused_preds:
			predictors = used_preds.copy()
			predictors.append(p)
			# print(predictors)
			# print(x_train)
			new_scores.append((p,apply_subset(x_train, y_train, x_test, y_test, ml_algorithm, predictors)))
		print(new_scores)
		new_scores.sort(key = lambda x: x[1][0])
		unused_preds.remove(new_scores[-1][0])
		used_preds.append(new_scores[-1][0])
		scores.append((used_preds.copy(),new_scores[-1][-1]))
		return forward_selection(x_train, y_train, x_test, y_test, ml_algorithm, unused_preds, used_preds, scores)





def constant_predictor(x_train, y_train, x_test, y_test):
	counts = get_counts(y_train)
	# print(counts)
	label = counts[-1][0]
	predictions = [label]*len(y_test)
	f1 = precision_recall_fscore_support(y_test, predictions, average = 'weighted')[2]
	accuracy = len(y_test[y_test == predictions])/len(y_test)
	print(precision_recall_fscore_support(y_test, predictions, average = 'weighted'))
	return(accuracy, f1)

def linear_predictor(x_train, y_train, x_test, y_test, give_clf = False):
	clf = LogisticRegression(solver = 'lbfgs', max_iter = 10000).fit(x_train, y_train)
	accuracy = clf.score(x_test, y_test)
	f1 = precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted')[2]
	print(precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted'))
	if not give_clf:
		return(accuracy, f1)
	else:
		return(clf)

def nn_predictor(x_train, y_train, x_test, y_test):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,7,7), random_state=1, max_iter = 10000)
	clf.fit(x_train, y_train)
	accuracy = clf.score(x_test, y_test)
	f1 = precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted')[2]
	return(accuracy, f1)

def rf_predictor(x_train, y_train, x_test, y_test, give_clf = False):
	clf = RandomForestClassifier(n_estimators =100)
	clf.fit(x_train, y_train)
	accuracy = clf.score(x_test, y_test)
	f1 = precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted')[2]
	print(precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted'))
	if not give_clf:
		return(accuracy, f1)
	else:
		return(clf)

def knn_predictor(x_train, y_train, x_test, y_test):
	clf = NearestNeighbors(n_neighbors = 5)
	clf.fit(x_train)
	accuracy = clf.score(x_test, y_test)
	f1 = precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted')[2]
	print(accuracy, f1)

def qda_predictor(x_train, y_train, x_test, y_test, give_clf = False):
	clf = QuadraticDiscriminantAnalysis()
	clf.fit(x_train, y_train)
	accuracy = clf.score(x_test, y_test)
	f1 = precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted')[2]
	print(precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted'))
	if not give_clf:
		return(accuracy, f1)
	else:
		return(clf)
	# return(clf)

def run_with_forward_selection(x_train, y_train, x_test, y_test):

	# print(get_counts(y_test))
	# # Number of `acquired` labels
	# # 338
	# # Number of `ipo` labels
	# # 63
	# # Number of `closed` labels
	# # 114
	# # Number of `operating` labels
	# # 3030

	# x_train = x_train.drop(y_train[y_train == 'operating'].index)
	# y_train = y_train.drop(y_train[y_train == 'operating'].index)
	# x_test = x_test.drop(y_test[y_test == 'operating'].index)
	# y_test = y_test.drop(y_test[y_test == 'operating'].index)
	# print(x_train.columns)

	# print('constant_predictor:')
	# print(constant_predictor(x_train, y_train, x_test, y_test))

	# print('linear_predictor:')
	# print(linear_predictor(x_train, y_train, x_test, y_test))

	# print('qda_predictor:')
	# print(qda_predictor(x_train, y_train, x_test, y_test))

	# print('rf_predictor:')
	# print(rf_predictor(x_train, y_train, x_test, y_test))

	# print('nn_predictor:')
	# print(nn_predictor(x_train, y_train, x_test, y_test))


	# Forward selection on predictors

	# print('\n linear_predictor')
	# predictor_scores = forward_selection(x_train, y_train, x_test, y_test, linear_predictor)
	# predictor_scores.sort(key = lambda x: x[1])
	# print(predictor_scores)

	print('\n linear_predictor')
	predictor_scores = forward_selection(x_train, y_train, x_test, y_test, rf_predictor)
	predictor_scores.sort(key = lambda x: x[1])
	print(predictor_scores)

	# print('\n qda_predictor')
	# predictor_scores = forward_selection(x_train, y_train, x_test, y_test, qda_predictor)
	# predictor_scores.sort(key = lambda x: x[1])
	# print(predictor_scores)

	# print('\n nn_predictor')
	# predictor_scores = forward_selection(x_train, y_train, x_test, y_test, nn_predictor)
	# predictor_scores.sort(key = lambda x: x[1])
	# print(predictor_scores)

def get_counts(y):
	counts = []
	# for label in ['acquired', 'ipo', 'closed', 'operating']:
	for label in set(y.values):
		counts.append((label, len(y[y == label])))
	counts.sort(key = lambda x: x[1])
	return(counts)

def sample_900(data):
	return util.sampling_k_elements(data, k = 900)

def sample_300(data):
	return util.sampling_k_elements(data, k = 300)

def main():
	csv_path = './data/crunchbase-companies.csv'

	ignored_preds = ['founded_quarter', 'country_code', 'category_code', 'region', 'city', 'state_code', 'name', 'permalink']
	# ignored_preds.extend(['city_importance', 'city_longitude', 'city_lattitude', 'state_importance', 'state_longitude', 'state_lattitude'])

	ignored_preds.extend([ 'city_longitude', 'city_lattitude', 'state_lattitude', 'city_importance'])
	ignored_preds.extend([ 'first_funding_at', 'last_funding_at',\
       'last_milestone_at'])
	train, valid, test = util.get_data(csv_path)
	# data = train.copy()
	# data.append(valid)
	# data.append(test)
	print(get_counts(train['status']))
	print(get_counts(valid['status']))
	print(get_counts(test['status']))
	# balancing data
	train_balanced = train.groupby('status').apply(sample_900).reset_index(drop=True)	
	perm = np.random.permutation(train_balanced.index)
	train_balanced = train_balanced.iloc[perm]

	valid_balanced = valid.groupby('status').apply(sample_300).reset_index(drop=True)	
	perm = np.random.permutation(valid_balanced.index)
	valid_balanced = valid_balanced.iloc[perm]

	# Splitting data into train/test sets
	x_train = train.loc[:, train.columns != 'status']
	x_train_balanced = train_balanced.loc[:, train_balanced.columns != 'status']

	y_train = train['status']
	y_train_balanced = train_balanced['status']
	y_train_01 = y_train.transform(lambda x: 0 if x != 'operating' else 1)

	x_valid = valid.loc[:, train.columns != 'status']
	x_valid_balanced = valid_balanced.loc[:, train_balanced.columns != 'status']

	y_valid = valid['status']
	y_valid_balanced = valid_balanced['status']
	y_valid_01 = y_valid.transform(lambda x: 0 if x != 'operating' else 1)

	y_test = test['status']
	x_test = test.loc[:, train.columns != 'status']
	y_test_01 = y_test.transform(lambda x: 0 if x != 'operating' else 1)
	

	# print(constant_predictor(x_train, y_train, x_test, y_test))

	# run_with_forward_selection(x_train, y_train, x_valid, y_valid)

	# predictors = ['last_funding_at_month', 'first_funding_at_month', 'funding_rounds', 'state_lattitude', 'state_importance', 'state_longitude', 'founded_at_month', 'last_milestone_at_month', 'last_milestone_at', 'founded_at_year', 'city_lattitude', 'funding_total_usd', 'last_milestone_at_year', 'founded_at']
	# print(apply_subset(x_train, y_train, x_train, y_train, rf_predictor, predictors))


	predictors = ['last_funding_at_month', 'first_funding_at_month', 'founded_at_month', 'last_milestone_at_month', 'state_longitude', 'city_importance']
	predictors2 = ['last_funding_at_year', 'first_funding_at_year', 'funding_rounds', 'state_importance', 'last_milestone_at', 'last_funding_at', 'founded_at', 'city_longitude', 'last_milestone_at_year', 'first_funding_at', 'funding_total_usd', 'founded_at_month', 'first_funding_at_month', 'founded_at_year', 'city_lattitude']
	predictors = ['last_funding_at_month', 'first_funding_at_month', 'founded_at_month', 'last_milestone_at_month', 'state_importance']
	predictors2 = ['last_funding_at_month', 'first_funding_at_month', 'funding_rounds', 'state_lattitude', 'state_importance', 'state_longitude', 'founded_at_month', 'last_milestone_at_month', 'last_milestone_at', 'founded_at_year', 'city_lattitude', 'funding_total_usd', 'last_milestone_at_year', 'founded_at']
	# scores = []
	# # for threshold in np.arange(0.05, 1, 0.05):
	threshold = .85

	# 	# print()
	# 	# print(threshold)
	x_valid = x_test
	y_valid_01 = y_test_01
	y_valid = y_test

	clf = apply_subset(x_train, y_train_01, x_valid, y_valid_01,  qda_predictor, predictors, give_clf = True)
	predicted = clf.predict_proba(x_valid[predictors])
	predicted = predicted[:,1]

	predicted= (predicted > threshold).astype('int')
	# predicted[:,1] = (predicted[:,1] >= threshold).astype('int')
	# print(predicted)
	# print(y_valid_01)
	# disp = confusion_matrix(y_valid_01, predicted)
	# # print(disp)

	# # Step 2
	
	train_predicted = clf.predict_proba(x_train[predictors])
	train_predicted = train_predicted[:,1]
	train_predicted = (train_predicted > threshold).astype('int')

	x_train_2 = x_train[train_predicted == 0]
	y_train_2 = y_train[train_predicted == 0]

	x_valid_2 = x_valid[predicted == 0]
	y_valid_2 = y_valid[predicted == 0]
	# # print(train_predicted)
	# # print(y_train_2)
	# predictors2 = forward_selection(x_train_2, y_train_2, x_valid_2, y_valid_2, rf_predictor)
	# predictors2 = predictors2[-1][0]
	clf = apply_subset(x_train_2, y_train_2, x_valid, y_valid,  rf_predictor, predictors2, give_clf = True)
	predicted_2 = clf.predict(x_valid[predictors2])


	predictions = []
	for(y1,y2) in zip(predicted, predicted_2):
		if y1 == 1:
			predictions.append('operating')
		else:
			predictions.append(y2)
	new_predictions = predictions
	# print(predictions)
	# print(y_valid)
	print(confusion_matrix(y_valid,predictions))
	df_cm = pd.DataFrame(array, index = [i for i in "ABCD"],
                  columns = [i for i in "ABCD"])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True)

	acc = len(y_valid[predictions == y_valid])/len(predictions)
	print(acc)
	print(precision_recall_fscore_support(y_valid, predictions, average = 'weighted'))

	# print(constant_predictor(x_train, y_train, x_valid, y_valid))
		# scores.append((threshold, acc))
	# scores.sort(key = lambda x: x[1])
	# print(scores)











	# Method: constant predictor
	# Accuracy = 0.8547249647	0.23272727272727273
	# print(constant_predictor(y_valid))
	# print(y_valid)



	# Method: QDA
	# predictors = 
	# print(best_subset(x_train, y_train, x_valid, y_valid, linear_predictor))

	# Method: Logistic Regression
	# Accuracy = 0.8524682651622003		0.5672727272727273
	# print(linear_predictor(x_train, y_train, x_valid, y_valid))

	# Method: Logistic Regression - good predictors only
	# # # Accuracy = 0.8550070521861777
	# train, valid, test = util.get_data(csv_path, good_preds_only = True, preprocess = True)
	 # (funding total usd, funding rounds)



	# print(best_subset(x_train, y_train, x_valid, y_valid,  qda))
	# predictors = ['state_lattitude', 'last_funding_at', 'funding_rounds', 'state_importance', 'city_importance', 'last_funding_at_year']



	# clf = qda(x_train[predictors], y_train, x_valid[predictors], y_valid)
	# print(x_valid)
	# disp = plot_confusion_matrix(clf, x_valid[predictors], y_valid)
	# disp.ax_.set_title('Confusion Matrix')
	# print(disp.confusion_matrix)
	# plt.show()
	# y_train = train['status']
	# x_train = train.loc[:, train.columns != 'status']

	# y_valid = valid['status']
	# x_valid = valid.loc[:, train.columns != 'status']
	# print(linear_predictor(x_train, y_train, x_valid, y_valid))

	# Method: Neural Network - (7,7,7)
	# Accuracy = 0.8555712270803949		0.6
	# print(nn_predictor(x_train, y_train, x_valid, y_valid))

	# Accuracy = 0.8496473906911143		0.6
	# print(rf_predictor(x_train, y_train, x_valid, y_valid))



main()