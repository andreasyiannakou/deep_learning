#for filtering files
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Activation
from keras.utils.generic_utils import get_custom_objects
from keras import models
print(keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import pickle
import datetime

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def srelu(x):
    if np.random.rand(1) > 0.5:
        return K.relu(x)
    return K.relu(x) - x

get_custom_objects().update({'srelu': Activation(srelu)})

def build_model(af, padding, pooling, kernel_size):
    model = models.Sequential()
    model.add(Conv2D(32, kernel_size, padding=padding, activation=af, input_shape=(150, 150, 3)))
    if pooling == 'Max':
        model.add(MaxPooling2D((2, 2)))
    if pooling == 'Avg':
        model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size, padding=padding, activation=af))
    if pooling == 'Max':
        model.add(MaxPooling2D((2, 2)))
    if pooling == 'Avg':
        model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size, padding=padding, activation=af))
    if pooling == 'Max':
        model.add(MaxPooling2D((2, 2)))
    if pooling == 'Avg':
        model.add(AveragePooling2D((2, 2)))       
    model.add(Conv2D(128, kernel_size, padding=padding, activation=af))
    model.add(Flatten())
    model.add(Dense(128, activation=af))
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
functions = ['relu', 'srelu']
padding = ['valid', 'same']
pooling = ['Max', 'Avg']
kernel = [(3,3), (5,5)]

d2 = datetime.datetime.now()
print(d2-d1)

num = 3
results = []

for i in range(num):
    for af in functions:
        for pad in padding:
            for pool in pooling:
                for size in kernel:
                    fn = str(af) + '_' + str(pad) + '_' + str(pool) + '_' + str(size[1]) + '.h5'
                    print(fn)
                    # build the model
                    try:
                        the_model = build_model(af, pad, pool, size)
                        history = the_model.fit_generator(train_generator, steps_per_epoch = 800, epochs=10, validation_data=validation_generator, validation_steps=200)
                        #append the results for further analysis
                        results.append([i, af, pad, pool, size, history.history['val_acc'][-1], history.history['val_loss'][-1], history.history['acc'], history.history['loss'], history.history['val_acc'], history.history['val_loss']])
                        print("CNN Error: %.2f%%" % (100-(history.history['val_acc'][-1])*100))
                        with open('CNN_results_initial.pkl', 'wb') as f:
                            pickle.dump(results, f)
                    except:
                        print('error - ', fn)

d3 = datetime.datetime.now()
print(d3-d2)