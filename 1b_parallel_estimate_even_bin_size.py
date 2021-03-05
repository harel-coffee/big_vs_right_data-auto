import json
import random
import multiprocessing as mp
import os
import time
import warnings

from copy import deepcopy

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ML_model import predict_class

warnings.filterwarnings(action="ignore", 
                        category=ConvergenceWarning)


suffix = '_compact'

target_name = 'spring_2015_cum_gpa'

# preprocess data
df = pd.read_pickle('data/preproc%s.pkl' % suffix)

# parse into features and target

target = 'spring_2015_cum_gpa'

X = df.drop(target,1)
y = df[target]

# convert target to categorical

edges = None

for bins in [3,4,5]:
	if edges is None:
		# Equal height binning
		y_class = pd.qcut(y,bins,labels=False)

	elif max(edges) < 1.01:
		y_class = pd.qcut(y,q=edges,labels=False)    

	else:
		y_class = pd.cut(y,edges,labels=False)


	# make splits 

	split_args = {'n_splits': 1000, 
				  'test_size': 1/4,
				  'random_state': 0}

	sss = StratifiedShuffleSplit(**split_args)
	splits = [[i]+list(s) for i, s in enumerate(sss.split(X,y_class))]
	splits_df = pd.DataFrame({'split_idx': range(split_args['n_splits'])})
	splits_df['merge_col'] = True    

	# export splits
	pd.DataFrame(data=splits, 
				 columns=['split_idx', 'train_idxs', 'test_idxs'])\
		.to_pickle('data/splits_even%i.pkl'%bins)


	# feature set definitionss
	with open('data/feature_sets%s.json' % suffix, 'r') as f:
		FS_dict = json.loads(f.read())

	FS_columns = ['feature_set_label', 'feature_set']
	FS_df = pd.Series(FS_dict)\
				.reset_index()\
				.rename(columns=dict(zip(['index',0],FS_columns)))
	FS_df['merge_col']=True

	# run experiments
	output_info = FS_df.merge(splits_df).drop('merge_col', 1)

	models = \
	[{'clf': 'LR', 'estimate_hyper_param':True, 'fs': True, 'clf_params': {'penalty': 'l2'}, 'poly_degree': 1}]


	for model_idx, model in enumerate(models[:]):
		
		output_file = 'model_output/model_%i_%ieven_bins%s.pkl' % (model_idx, bins, suffix)

		#if not os.path.exists(output_file):

		jobs = [(X, y_class, fs, split, model) for fs in FS_dict.items() for split in splits]

		output_cols = ['feature_set_label', 'split_idx', 
					   'accuracy', 'y_pred', 'param_opt']

		print('Start multiprocessing for specfication:\n', model)
		random.shuffle(jobs)

		starttime = time.time()    
		

		output = Parallel(n_jobs=20)(map(delayed(predict_class), jobs))
		output_df = output_info.merge(pd.DataFrame(output, columns=output_cols))

		for k,v in model.items():
			if k == 'clf_params':
				for k_clf, v_clf in model['clf_params'].items():
					output_df['clf_'+k_clf] = v_clf
			else:
				output_df[k] = v
		output_df.to_pickle(output_file)

		print('Completed model with above specfication.')
		print('Completion time (s):', round(time.time()-starttime, 1),'\n')

		print(output_df.groupby(['feature_set_label']).accuracy.mean())
		