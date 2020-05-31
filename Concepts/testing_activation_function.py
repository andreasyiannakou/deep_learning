	
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
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
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

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
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation=af))
	model.add(Flatten())
	model.add(Dense(128, activation=af))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

activation_functions = ['relu', 'srelu', 'elu', 'selu', 'sigmoid', 'tanh', 'linear']
num = 10
results = []

for i in range(num):
    for af in activation_functions:
        # build the model
        model = baseline_model(af)
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

for row in results_new:
    row_vals = pd.DataFrame([row[1], row[2], row[3]]).T
    row_vals.columns = ['Name', 'Loss', 'Acc']
    row_vals[['Loss', 'Acc']] = row_vals[['Loss', 'Acc']].astype(float)
    acc = acc.append(row_vals)

result_mean = acc.groupby('Name').mean()
result_std = acc.groupby('Name').std()
result_max = acc.groupby('Name').max()
result_min = acc.groupby('Name').min()

#Best ReLU = 0, best SReLU = 64

import matplotlib.pyplot as plt
"""
#ReLU loss graph
loss_r = results_new[0][5]
val_loss_r = results_new[0][7]
epochs = range(1,11)
plt.plot(epochs, loss_r, 'bo', label='Training loss')
plt.plot(epochs, val_loss_r, 'b', label='Validation loss')
plt.title('Training and validation loss for best ReLU accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#SReLU loss graph
loss_sr = results_new[64][5]
val_loss_sr = results_new[64][7]
epochs = range(1,11)
plt.plot(epochs, loss_sr, 'bo', label='Training loss')
plt.plot(epochs, val_loss_sr, 'b', label='Validation loss')
plt.title('Training and validation loss for best SReLU accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_r = results_new[0][4]
val_acc_r = results_new[0][6]
epochs = range(1,11)
plt.plot(epochs, acc_r, 'bo', label='Training acc')
plt.plot(epochs, val_acc_r, 'b', label='Validation acc')
plt.title('Training and validation accuracy for best ReLU accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

acc_sr = results_new[64][4]
val_acc_sr = results_new[64][6]
epochs = range(1,11)
plt.plot(epochs, acc_sr, 'bo', label='Training acc')
plt.plot(epochs, val_acc_sr, 'b', label='Validation acc')
plt.title('Training and validation accuracy for best SReLU accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

f, axarr = plt.subplots(2, sharex=True)
#ReLU loss graph
acc_r = results_new[0][4]
val_acc_r = results_new[0][6]
epochs = range(1,11)
axarr[0].plot(epochs, acc_r, 'bo', label='Training accuracy')
axarr[0].plot(epochs, val_acc_r, 'b', label='Validation accuracy')
axarr[0].set_title('Training and validation accuracy and loss for best ReLU')
axarr[0].set_ylabel('Accuracy')
axarr[0].legend()

loss_r = results_new[0][5]
val_loss_r = results_new[0][7]
epochs = range(1,11)
axarr[1].plot(epochs, loss_r, 'bo', label='Training loss')
axarr[1].plot(epochs, val_loss_r, 'b', label='Validation loss')
axarr[1].set_xlabel('Epochs')
axarr[1].set_ylabel('Loss')
axarr[1].legend()

f, axarr = plt.subplots(2, sharex=True)
#SReLU loss graph
acc_sr = results_new[64][4]
val_acc_sr = results_new[64][6]
epochs = range(1,11)
axarr[0].plot(epochs, acc_sr, 'bo', label='Training accuracy')
axarr[0].plot(epochs, val_acc_sr, 'b', label='Validation accuracy')
axarr[0].set_title('Training and validation accuracy and loss for best SReLU')
axarr[0].set_ylabel('Accuracy')
axarr[0].legend()

loss_sr = results_new[64][5]
val_loss_sr = results_new[64][7]
epochs = range(1,11)
axarr[1].plot(epochs, loss_sr, 'bo', label='Training loss')
axarr[1].plot(epochs, val_loss_sr, 'b', label='Validation loss')
axarr[1].set_xlabel('Epochs')
axarr[1].set_ylabel('Loss')
axarr[1].legend()


f, axarr = plt.subplots(2, 2)
f.set_figwidth(16)
f.set_figheight(6)
#ReLU loss graph
acc_r = results_new[0][4]
val_acc_r = results_new[0][6]
epochs = range(1,11)
axarr[0,0].plot(epochs, acc_r, 'bo', label='Training accuracy')
axarr[0,0].plot(epochs, val_acc_r, 'b', label='Validation accuracy')
axarr[0,0].set_title('Training and validation accuracy and loss for best ReLU')
axarr[0,0].set_ylabel('Accuracy')
axarr[0,0].legend()

loss_r = results_new[0][5]
val_loss_r = results_new[0][7]
epochs = range(1,11)
axarr[1,0].plot(epochs, loss_r, 'bo', label='Training loss')
axarr[1,0].plot(epochs, val_loss_r, 'b', label='Validation loss')
axarr[1,0].set_xlabel('Epochs')
axarr[1,0].set_ylabel('Loss')
axarr[1,0].legend()

#SReLU loss graph
acc_sr = results_new[64][4]
val_acc_sr = results_new[64][6]
epochs = range(1,11)
axarr[0,1].plot(epochs, acc_sr, 'bo', label='Training accuracy')
axarr[0,1].plot(epochs, val_acc_sr, 'b', label='Validation accuracy')
axarr[0,1].set_title('Training and validation accuracy and loss for best SReLU')
axarr[0,1].set_ylabel('Accuracy')
axarr[0,1].legend()

loss_sr = results_new[64][5]
val_loss_sr = results_new[64][7]
epochs = range(1,11)
axarr[1,1].plot(epochs, loss_sr, 'bo', label='Training loss')
axarr[1,1].plot(epochs, val_loss_sr, 'b', label='Validation loss')
axarr[1,1].set_xlabel('Epochs')
axarr[1,1].set_ylabel('Loss')
axarr[1,1].legend()
"""


