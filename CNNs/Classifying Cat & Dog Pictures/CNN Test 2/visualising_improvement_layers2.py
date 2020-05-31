import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import datetime
d1 = datetime.datetime.now()

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

# build the model
model2 = build_model(0, 'No', 128)

model2.load_weights(r"C:\Users\Andreas\Documents\GitHub\DeepLearning\CNNs\CNN Test 2\0_No_128.h5")

layer_outputs = [layer.output for layer in model2.layers]
activation_model = models.Model(inputs=model2.input, outputs=layer_outputs)

img = image.load_img(r"C:\Users\Andreas\Documents\GitHub\DeepLearning\CNNs\CNN Test 2\dog.4.jpg", target_size=(150,150,3))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.

plt.imshow(img_array[0])

activations = activation_model.predict(img_array)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model2.layers[:7]:
    layer_names.append(layer.name)


images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    print(layer_activation.shape)
    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='gray')
    
plt.show()
