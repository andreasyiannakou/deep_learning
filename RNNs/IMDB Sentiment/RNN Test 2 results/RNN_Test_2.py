#Import packages
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional
from sklearn.model_selection import train_test_split

import numpy as np
import datetime

import pickle

d1 = datetime.datetime.now()
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Number of words to consider as features
max_features = 10000
# Cut texts after this number of words (among top max_features most common words)
maxlen = 500
# Load the data as lists of integers.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X = np.append(X_train, X_test)
y = np.append(y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
# This turns our lists of integers into a 2D integer tensor of shape `(samples, maxlen)`
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

# build the different models
def build_model(name, max_features, maxlen, base_model):
    model = Sequential()
    model.add(Embedding(max_features, 32, input_length=maxlen))
    model.set_weights(base_model.get_weights())
    if name == 'LSTM_RD':
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    elif name == 'GRU_RD':
        model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
    elif name == 'BiDi_RD':
        model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2))) 
    else:
        return None
    
    # We add the classifier on top
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

d2 = datetime.datetime.now()
print(d2-d1)

# the _RD refers to recurrent dropout, BiDi refers to bidirectional models
options = ['LSTM_RD', 'GRU_RD', 'BiDi_RD']

num = 3
results = []

base_model = Sequential()
# We specify the maximum input length to our Embedding layer so we can later flatten the embedded inputs
base_model.add(Embedding(max_features, 32, input_length=maxlen))

for i in range(num):
    for option in options:
        # build the model
        model = build_model(option, max_features, maxlen, base_model)
        if model is None:
            print('No model')
            continue
        else:
            # Fit the model
            history = model.fit(X_train, y_train, epochs=10, batch_size=200, validation_data=(X_test, y_test))
            #append the results for further analysis
            results.append([i, option, history.history['val_acc'][-1], history.history['val_loss'][-1], history.history['acc'], history.history['loss'], history.history['val_acc'], history.history['val_loss']])
            print("RNN Error: %.2f%%" % (100-(history.history['val_acc'][-1])*100))
        with open('RNN_results_2.pkl', 'wb') as f:
            pickle.dump(results, f)

d3 = datetime.datetime.now()
print(d3-d2)