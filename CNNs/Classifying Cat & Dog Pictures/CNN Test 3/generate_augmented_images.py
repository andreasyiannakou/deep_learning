from keras import models
from keras.preprocessing.image import ImageDataGenerator

# set up the generators
rotate_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=45)
rotate_generator = rotate_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_rotate", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

width_datagen = ImageDataGenerator(rescale = 1./255, width_shift_range=0.25)
width_generator = width_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_width", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

height_datagen = ImageDataGenerator(rescale = 1./255, height_shift_range=0.25)
height_generator = height_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_height", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

shear_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.25)
shear_generator = shear_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_shear", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

zoom_datagen = ImageDataGenerator(rescale = 1./255, zoom_range=0.25)
zoom_generator = zoom_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_zoom", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

horizontal_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
horizontal_generator = horizontal_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_horizontal", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

augmented_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, shear_range=0.25, zoom_range=0.25, horizontal_flip=True)
augmented_generator = augmented_datagen.flow_from_directory("./invariance_no_augmentation", save_to_dir="./invariance_augmented", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

# load trained model
model = models.load_model('1_30_aug.h5')

# generate the testing data
model.evaluate_generator(rotate_generator, steps=1)
model.evaluate_generator(width_generator, steps=1)
model.evaluate_generator(height_generator, steps=1)
model.evaluate_generator(shear_generator, steps=1)
model.evaluate_generator(zoom_generator, steps=1)
model.evaluate_generator(horizontal_generator, steps=1)
model.evaluate_generator(augmented_generator, steps=1)
