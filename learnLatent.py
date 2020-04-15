import os
import pickle
import config
import dnnlib
import gzip
import json
import numpy as np
from tqdm import tqdm_notebook

import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
    
# with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
with open('data/latent_training_data.pkl.gz', 'rb') as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

# labels_data[0]  # wow. many fields. amaze
# {'faceId': 'b6807d9a-0ab5-4595-9037-c69c656c5c38',
#  'faceRectangle': {'top': 322, 'left': 223, 'width': 584, 'height': 584},
#  'faceLandmarks': {'pupilLeft': {'x': 386.0, 'y': 480.7},
#   ...
#   'underLipBottom': {'x': 525.5, 'y': 800.8}},
#  'faceAttributes': {'smile': 0.999,
#   'headPose': {'pitch': 0.0, 'roll': -0.4, 'yaw': 3.1},
#   'gender': 'male',
#   'age': 50.0,
#   ...
#   }}

# Let's play with age and gender
# you can train your own model now
X_data = dlatent_data.reshape((-1, 18*512))
y_yaw_data = np.array([x['faceAttributes']['headPose']['yaw'] for x in labels_data])
young_idx = [i for i in range(len(labels_data)) if labels_data[i]['faceAttributes']['age'] < 15]
X_young_data = dlatent_data[young_idx, :, :].reshape((-1, 18*512))
y_yaw_young_data = np.array([labels_data[i]['faceAttributes']['headPose']['yaw'] for i in young_idx])

assert(len(X_data) == len(y_yaw_data))
assert(len(X_young_data) == len(y_yaw_young_data))
print(len(X_data))
print(len(X_young_data))

from sklearn import linear_model
reg = linear_model.LinearRegression()
val_scores = cross_val_score(reg, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
print('val_scores:', val_scores)
print(np.mean(val_scores))

reg_young = linear_model.LinearRegression()
val_young_scores = cross_val_score(reg_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
print('val_young_scores:', val_young_scores)
print(np.mean(val_young_scores))

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_val_scores = cross_val_score(poly, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
print('poly_val_scores:', poly_val_scores)
print(np.mean(poly_val_scores))

poly_young = PolynomialFeatures(degree=2)
poly_val_young_scores = cross_val_score(poly_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
print('poly_val_young_scores:', poly_val_young_scores)
print(np.mean(poly_val_young_scores))

# # So let's find the gender direction in the latent space
# clf = LogisticRegression(class_weight='balanced').fit(X_data.reshape((-1, 18*512)), y_gender_data)
# gender_direction = clf.coef_.reshape((18, 512))

# # Evaluation
# clf = SGDClassifier('log', class_weight='balanced') # SGB model for performance sake
# scores = cross_val_score(clf, X_data, y_gender_data, scoring='accuracy', cv=5)
# clf.fit(X_data, y_gender_data)

# print(scores)
# print('Mean: ', np.mean(scores))