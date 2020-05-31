from keras import models
from keras.preprocessing.image import ImageDataGenerator
import datetime

d1 = datetime.datetime.now()

test_datagen = ImageDataGenerator(rescale = 1./255)
no_aug_generator = test_datagen.flow_from_directory("./invariance_no_augmentation", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
rotate_generator = test_datagen.flow_from_directory("./invariance_rotate", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
width_generator = test_datagen.flow_from_directory("./invariance_width", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
height_generator = test_datagen.flow_from_directory("./invariance_height", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
shear_generator = test_datagen.flow_from_directory("./invariance_shear", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
zoom_generator = test_datagen.flow_from_directory("./invariance_zoom", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
horizontal_generator = test_datagen.flow_from_directory("./invariance_horizontal", target_size = (150, 150), batch_size = 20, class_mode = 'binary')
augmented_generator = test_datagen.flow_from_directory("./invariance_augmented", target_size = (150, 150), batch_size = 20, class_mode = 'binary')

d2 = datetime.datetime.now()
print(d2-d1)

# load teh augmented model
model_aug = models.load_model('1_30_aug.h5')

# generate the testing data for augmented model
eval_no_aug1 = model_aug.evaluate_generator(no_aug_generator, steps=1)
eval_rotate1 = model_aug.evaluate_generator(rotate_generator, steps=1)
eval_width1 = model_aug.evaluate_generator(width_generator, steps=1)
eval_height1 = model_aug.evaluate_generator(height_generator, steps=1)
eval_shear1 = model_aug.evaluate_generator(shear_generator, steps=1)
eval_zoom1 = model_aug.evaluate_generator(zoom_generator, steps=1)
eval_horizontal1 = model_aug.evaluate_generator(horizontal_generator, steps=1)
eval_augmented1 = model_aug.evaluate_generator(augmented_generator, steps=1)

# reset generators
no_aug_generator.reset()
rotate_generator.reset()
width_generator.reset()
height_generator.reset()
shear_generator.reset()
zoom_generator.reset()
horizontal_generator.reset()
augmented_generator.reset()

# load teh augmented model
model_no_aug = models.load_model('2_20_no_aug.h5')

# generate the testing data for augmented model
eval_no_aug2 = model_no_aug.evaluate_generator(no_aug_generator, steps=1)
eval_rotate2 = model_no_aug.evaluate_generator(rotate_generator, steps=1)
eval_width2 = model_no_aug.evaluate_generator(width_generator, steps=1)
eval_height2 = model_no_aug.evaluate_generator(height_generator, steps=1)
eval_shear2 = model_no_aug.evaluate_generator(shear_generator, steps=1)
eval_zoom2 = model_no_aug.evaluate_generator(zoom_generator, steps=1)
eval_horizontal2 = model_no_aug.evaluate_generator(horizontal_generator, steps=1)
eval_augmented2 = model_no_aug.evaluate_generator(augmented_generator, steps=1)

d2 = datetime.datetime.now()
print(d2-d1)