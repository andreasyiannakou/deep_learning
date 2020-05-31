#for filtering files
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import models
print(keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
import pickle
import datetime

d1 = datetime.datetime.now()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def build_model():
    model = models.Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(150, 150, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))       
    model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

train_datagen = ImageDataGenerator(rescale = 1./255)
aug_train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, shear_range=0.25, zoom_range=0.25, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory("./train", target_size = (150,150), batch_size = 25, class_mode = 'binary')
aug_train_generator = aug_train_datagen.flow_from_directory("./train", target_size = (150,150), batch_size = 25, class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory("./test", target_size = (150, 150), batch_size = 25, class_mode = 'binary')

epochs = [25, 30]
training_generator = [(train_generator, 'no_aug'), (aug_train_generator, 'aug')]

d2 = datetime.datetime.now()
print(d2-d1)

num = 3
results = []

for i in range(num):
        for epoch in epochs:
            for tg, aug in training_generator:
                try:
                    fn = str(i) + '_' + str(epoch) + '_' + str(aug) + '.h5'
                    print(fn)
                    # build the model
                    the_model = build_model()
                    history = the_model.fit_generator(tg, steps_per_epoch = 800, epochs=epoch, validation_data=validation_generator, validation_steps=200)
                    #append the results for further analysis
                    results.append([i, epoch, aug, history.history['val_acc'][-1], history.history['val_loss'][-1], history.history['acc'], history.history['loss'], history.history['val_acc'], history.history['val_loss']])
                    print("CNN Error: %.2f%%" % (100-(history.history['val_acc'][-1])*100))
                    with open('CNN_results_data_aug2.pkl', 'wb') as f:
                        pickle.dump(results, f)
                    try:
                        the_model.save(fn)
                    except:
                        print('save failed')
                except:
                    print('failed - ', fn)

d3 = datetime.datetime.now()
print(d3-d2)