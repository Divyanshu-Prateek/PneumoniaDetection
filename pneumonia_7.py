# -*- coding: utf-8 -*-
"""
Created on Fri May 29 00:10:42 2020

@author: prate
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:13:51 2020

@author: prate
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (96, 96, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
Dropout(0.2)
# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
Dropout(0.5)
# Step 4 - Full connection
classifier.add(Dense(units = 96, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 30,
                                   horizontal_flip = False,
                                   vertical_flip=False)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'G:\Internship\dataset\train',
                                                 target_size = (96, 96),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'G:\Internship\dataset\test',
                                            target_size = (96, 96),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 163,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 20)
classifier.save('Model_pneumonia7.h5')
# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'G:\Internship\dataset\val\NORMAL\NORMAL2-IM-1431-0001.jpeg', target_size = (96, 96))
test_image = image.img_to_array(test_image)
test_image = ImageDataGenerator(rescale = 1./255)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'Normal'
else:
    prediction = 'Pneumonia'
print(prediction)
