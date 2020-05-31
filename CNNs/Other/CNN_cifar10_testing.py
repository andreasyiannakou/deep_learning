import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import datetime
d1 = datetime.datetime.now()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#custom activation
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def srelu(x):
    if np.random.rand(1) > 0.5:
        return K.relu(x)
    return K.relu(x) - x

get_custom_objects().update({'srelu': Activation(srelu)})

def baseline_model(af):
    # create model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512, activation=af))
    model.add(Dense(128, activation=af))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def single_convolution_model(af):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(Flatten())
    model.add(Dense(128, activation=af))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def single_convolution_and_pooling(af):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=af))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def double_convolution_and_pooling(af):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(Flatten())
    model.add(Dense(128, activation=af))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def triple_convolution_and_pooling(af):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(Flatten())
    model.add(Dense(128, activation=af))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def triple_convolution_and_pooling_and_dropout(af):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=af))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation=af))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

activation_functions = ['relu', 'srelu']
mdl_list = [baseline_model, single_convolution_model, single_convolution_and_pooling, double_convolution_and_pooling, triple_convolution_and_pooling, triple_convolution_and_pooling_and_dropout]
num = 1
results = []

for i in range(num):
    for mdl in mdl_list:
        for af in activation_functions:
            # build the model
            model = mdl(af)
            # Fit the model
            history = model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=(1/12), verbose=0)
            # Final evaluation of the model
            scores = model.evaluate(X_test, y_test, verbose=0)
            results.append([i, af, scores[0], scores[1], history.history['acc'], history.history['loss'], history.history['val_acc'], history.history['val_loss']])
            print("CNN Error: %.2f%%" % (100-scores[1]*100))

d2 = datetime.datetime.now()
print(d2-d1)
"""
import pickle
with open('activation_function_results.pkl', 'wb') as f:
    pickle.dump(results, f)
"""
"""
with open('activation_function_results.pkl', 'rb') as f:
    results_new = pickle.load(f)
"""

import pandas as pd
acc = pd.DataFrame(columns=['Name', 'Loss', 'Acc'])

for row in results:
    row_vals = pd.DataFrame([row[1], row[2], row[3]]).T
    row_vals.columns = ['Name', 'Loss', 'Acc']
    row_vals[['Loss', 'Acc']] = row_vals[['Loss', 'Acc']].astype(float)
    acc = acc.append(row_vals)

result_mean = acc.groupby('Name').mean()
result_std = acc.groupby('Name').std()
result_max = acc.groupby('Name').max()
result_min = acc.groupby('Name').min()
