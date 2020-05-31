
import numpy as np
import pandas as pd
from keras.datasets import boston_housing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
from keras import backend as K
import datetime

#custom activation
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def srelu(x):
    if np.random.rand(1) > 0.5:
        return K.relu(x)
    return K.relu(x) - x

get_custom_objects().update({'srelu': Activation(srelu)})

# load dataset
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# define base model
def baseline_model(af):
    # create model
    model = Sequential()
    model.add(Dense(1024, input_shape=(train_data.shape[1],), kernel_initializer='normal', activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(1024, activation=af))
    model.add(Dense(256, activation=af))
    model.add(Dense(64, activation=af))
    model.add(Dense(16, activation=af))
    model.add(Dense(4, activation=af))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
   
d1 = datetime.datetime.now()

# build the model
model = baseline_model('srelu')
w1 = model.get_weights()
w10 = w1[0]
print(model.summary())

# Fit the model
model.fit(train_data, train_labels, validation_split=0.1, epochs=20, batch_size=200, verbose=0)
w2 = model.get_weights()
w20 = w2[0]

# Final evaluation of the model
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))


d2 = datetime.datetime.now()
print(d2-d1)

wd = w20 - w10
d = wd.sum()
print(d)


"""
# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""