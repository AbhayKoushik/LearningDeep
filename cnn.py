# Convolutional Neural Network

# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier=Sequential()

# Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu')) # input_shape in theano background will be (3,64,64)

# Pooling
classifier.add(MaxPooling2D(pool_size=(2,2))) # 2x2 is optimal

# Adding a second convolutional layer to improve the CNN
classifier.add(Convolution2D(32,(3,3),activation='relu')) # input_shape in theano background will be (3,64,64)
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,(3,3),activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,(3,3),activation='relu')) 

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(activation='relu',units=128))
classifier.add(Dropout(0.3))    
classifier.add(Dense(activation='sigmoid',units=1))  # output units

# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('dataset/training_set',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='binary')

test_set=test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

# making a new prediction
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'

    