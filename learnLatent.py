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
from sklearn.neural_network import MLPRegressor

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

mlp = MLPRegressor((512,10))
mlp_val_scores = cross_val_score(mlp, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
print('512_10_mlp_val_scores:', mlp_val_scores)
print(np.mean(mlp_val_scores))

mlp_young = MLPRegressor((512,10))
mlp_val_young_scores = cross_val_score(mlp_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
print('512_10_mlp_val_young_scores:', mlp_val_young_scores)
print(np.mean(mlp_val_young_scores))

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# works bit better, but in general accuracy is quite similar to the linear model
model.fit(X_data.reshape((-1, 18*512)), y_gender_data, validation_split=0.2, epochs=5)
model = Model(model.input, model.layers[-2].output)

# some dark magic is happening here
embedding_model = Sequential()
embedding_model.add(Embedding(10, 18*512, input_length=1)) # it's actually just a variable
# 10 is the input range
embedding_model.add(Flatten())

nonliner_gender_model = Model(embedding_model.input, model(embedding_model.output))
nonliner_gender_model.layers[-1].trainable = False # fix non-linear model and train only embeddings
nonliner_gender_model.compile('sgd', 'mse')

nonliner_gender_model.layers[1].set_weights([X_data[:10].reshape((-1, 18*512))])
y_data_real = nonliner_gender_model.predict(np.arange(10))

# and here
nonliner_gender_model.fit(np.arange(10), np.full((10, 1), 20), verbose=0, epochs=500)
nonliner_gender_model.predict(np.arange(10))
for v in embedding_model.layers[0].get_weights()[0]:
    plt.imshow(generate_image(v))
    plt.show()

# reset latents and try it over but now in another direction 
nonliner_gender_model.layers[1].set_weights([X_data[:10].reshape((-1, 18*512))])

nonliner_gender_model.fit(np.arange(10), np.full((10, 1), -20), verbose=0, epochs=500)

for v in embedding_model.layers[0].get_weights()[0]:
    plt.imshow(generate_image(v))
    plt.show()

# mlp = MLPRegressor((512,20))
# mlp_val_scores = cross_val_score(mlp, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
# print('512_20_mlp_val_scores:', mlp_val_scores)
# print(np.mean(mlp_val_scores))

# mlp_young = MLPRegressor((512,20))
# mlp_val_young_scores = cross_val_score(mlp_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
# print('512_20_mlp_val_young_scores:', mlp_val_young_scores)
# print(np.mean(mlp_val_young_scores))

# from sklearn import linear_model
# reg = linear_model.LinearRegression()
# val_scores = cross_val_score(reg, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
# print('val_scores:', val_scores)
# print(np.mean(val_scores))

# reg_young = linear_model.LinearRegression()
# val_young_scores = cross_val_score(reg_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
# print('val_young_scores:', val_young_scores)
# print(np.mean(val_young_scores))

# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor()
# mlp_val_scores = cross_val_score(mlp, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
# print('mlp_val_scores:', mlp_val_scores)
# print(np.mean(mlp_val_scores))

# mlp_young = MLPRegressor()
# mlp_val_young_scores = cross_val_score(mlp_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
# print('mlp_val_young_scores:', mlp_val_young_scores)
# print(np.mean(mlp_val_young_scores))

# mlp = MLPRegressor((512,))
# mlp_val_scores = cross_val_score(mlp, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
# print('512_mlp_val_scores:', mlp_val_scores)
# print(np.mean(mlp_val_scores))

# mlp_young = MLPRegressor((512,))
# mlp_val_young_scores = cross_val_score(mlp_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
# print('512_mlp_val_young_scores:', mlp_val_young_scores)
# print(np.mean(mlp_val_young_scores))

# mlp = MLPRegressor((512,32))
# mlp_val_scores = cross_val_score(mlp, X_data, y_yaw_data, scoring='neg_mean_squared_error', cv=5)
# print('512_32_mlp_val_scores:', mlp_val_scores)
# print(np.mean(mlp_val_scores))

# mlp_young = MLPRegressor((512,32))
# mlp_val_young_scores = cross_val_score(mlp_young, X_young_data, y_yaw_young_data, scoring='neg_mean_squared_error', cv=5)
# print('512_32_mlp_val_young_scores:', mlp_val_young_scores)
# print(np.mean(mlp_val_young_scores))

# # So let's find the gender direction in the latent space
# clf = LogisticRegression(class_weight='balanced').fit(X_data.reshape((-1, 18*512)), y_gender_data)
# gender_direction = clf.coef_.reshape((18, 512))

# # Evaluation
# clf = SGDClassifier('log', class_weight='balanced') # SGB model for performance sake
# scores = cross_val_score(clf, X_data, y_gender_data, scoring='accuracy', cv=5)
# clf.fit(X_data, y_gender_data)

# print(scores)
# print('Mean: ', np.mean(scores))