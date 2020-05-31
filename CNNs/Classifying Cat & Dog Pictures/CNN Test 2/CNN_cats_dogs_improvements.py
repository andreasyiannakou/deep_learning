#for filtering files
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import models
print(keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
import pickle
import datetime

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def build_model(dropout, include_BN, neurons):
    model = models.Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(150, 150, 3)))
    if include_BN == 'Yes':
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
    if include_BN == 'Yes':
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
    if include_BN == 'Yes':
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))       
    model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
    if include_BN == 'Yes':
        model.add(BatchNormalization())
    model.add(Flatten())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(neurons, activation='relu'))
    if include_BN == 'Yes':
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

d1 = datetime.datetime.now()

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory("./train", target_size = (150,150), batch_size = 25, class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory("./test", target_size = (150, 150), batch_size = 25, class_mode = 'binary')

# the _RD refers to recurrent dropout, BiDi refers to bidirectional models
dropout_rates = [0, 0.5]
include_BatchNormalisation = ['Yes', 'No']
dense_neurons = [128, 256]

d2 = datetime.datetime.now()
print(d2-d1)

num = 3
results = []

for i in range(num):
        for dropout in dropout_rates:
            for BN in include_BatchNormalisation:
                for dneurons in dense_neurons:
                    try:
                        fn = str(dropout) + '_' + str(BN) + '_' + str(dneurons) + '.h5'
                        print(fn)
                        # build the model
                        
                        the_model = build_model(dropout, BN, dneurons)
                        history = the_model.fit_generator(train_generator, steps_per_epoch = 800, epochs=10, validation_data=validation_generator, validation_steps=200)
                        #append the results for further analysis
                        results.append([i, dropout, BN, dneurons, history.history['val_acc'][-1], history.history['val_loss'][-1], history.history['acc'], history.history['loss'], history.history['val_acc'], history.history['val_loss']])
                        print("CNN Error: %.2f%%" % (100-(history.history['val_acc'][-1])*100))
                        with open('CNN_results_improvements.pkl', 'wb') as f:
                            pickle.dump(results, f)
                        try:
                            the_model.save(fn)
                        except:
                            print('save failed')
                    except:
                        print('failed - ', fn)

d3 = datetime.datetime.now()
print(d3-d2)