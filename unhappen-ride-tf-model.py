import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import read_raw, plot_history
import os
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from pathlib import Path

config = read_raw('config.cfg')


WIDTH = round(config['trainning'].getint('width'))
HEIGHT = round(config['trainning'].getint('height'))

IMG_SHAPE = (WIDTH, HEIGHT, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#base_model = InceptionV3(input_shape = IMG_SHAPE,
                                include_top = False,
                                weights='imagenet')
#for layer in base_model.layers:
#    layer.trainable = False
#    layer.trainable = False

base_model.summary()
base_model.trainable=False
#last_layer = pre_trained_model.get_layer('mixed7')
#print('last layer output shape: ', last_layer.output_shape)
#last_output = last_layer.output

#last_layer = base_model.get_layer('mixed7')
#print('last layer output shape: ', last_layer.output_shape)
#last_output = last_layer.output


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  tf.keras.layers.Dropout(0.2),
  prediction_layer,
])

#base_dir = '/Users/giuseppemarotta/Documents/raw-data/project-x'
SOURCEDIR = Path()

TRAINING_UNHAPPEN_DIR = "tmp/unhappen-v-happen/training/unhappen/"
TESTING_UNHAPPEN_DIR = "tmp/unhappen-v-happen/testing/unhappen/"
TRAINING_HAPPEN_DIR = "tmp/unhappen-v-happen/training/happen/"
TESTING_HAPPEN_DIR = "tmp/unhappen-v-happen/testing/happen/"

#train_dir = os.path.join(base_dir, 'train')
train_dir = "tmp/unhappen-v-happen/training/"
validation_dir = "tmp/unhappen-v-happen/testing/"



#validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
#train_valid_rides_dir = os.path.join(train_dir, 'valid')

# Directory with our training dog pictures
#train_invalid_rides_dir = os.path.join(train_dir, 'invalid')

# Directory with our training cat pictures
#validation_valid_rides_dir = os.path.join(validation_dir, 'valid')

# Directory with our training dog pictures
#validation_invalid_rides_dir = os.path.join(validation_dir, 'invalid')

train_datagen = ImageDataGenerator( rescale =1.0/255.,
                                    fill_mode='constant',
                                    width_shift_range=0.2,
                                    zoom_range=0.1,
                                    cval=255
                                    )

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(WIDTH, HEIGHT),  # All images will be resized to 150x150
        color_mode='rgb',
        class_mode='binary',
        shuffle=True,
        )

validation_datagen = ImageDataGenerator(rescale=1.0/255.)

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # This is the source directory for training images
        target_size=(WIDTH, HEIGHT),  # All images will be resized to 150x150
        batch_size=16,
        color_mode='rgb',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary',
        shuffle=True)

# All images will be rescaled by 1./255


checkpoint_path = "uh_training_retrain/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=1)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              #optimizer=SGD(lr=0.1, momentum=0.9),
              metrics=['acc'])
print('Print in ' + str(STEP_SIZE_TRAIN))
model.summary()
print('Number of Steps =' +str(STEP_SIZE_TRAIN))
history = model.fit_generator(
      train_generator,
      validation_data=validation_generator,
      epochs=5,
      verbose=1,
      callbacks=[cp_callback],
      use_multiprocessing=False
)

plot_history(history)