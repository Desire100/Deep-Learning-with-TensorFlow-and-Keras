# Convolutional Neural Networks for Image Classification

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualizing the Image Data

import matplotlib.pyplot as plt

x_train.shape
single_image = x_train[0]
single_image
single_image.shape
plt.imshow(single_image)

# Preprocessing

"""
We first need to make sure the labels will be understandable by our CNN.

""""
# Labels

y_train
y_test

"""
Hmmm, looks like our labels are literally categories of numbers. 
We need to translate this to be "one hot encoded" so our CNN can understand,
otherwise it will think this is some sort of regression problem on a
continuous axis.Luckily ,Keras has an easy to use function for this:
    
"""

from keras.utils.np_utils import to_categorical

y_train.shape
y_example = to_categorical(y_train)
y_example
y_example.shape
y_example[0]

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

#  Processing X Data
# We should normalize the X data

single_image.max()
single_image.min()

x_train = x_train/255
x_test = x_test/255

scaled_single = x_train[0]
scaled_single.max() 
plt.imshow(scaled_single, cmap='gray_r')

# Reshaping the Data
""""
Right now our data is 60,000 images stored in 28 by 28 pixel array formation. 

This is correct for a CNN, but we need to add one more dimension
 to show we're dealing with 1 RGB channel (since technically 
 the images are in black and white, only showing values from 0-255 
 on a single channel), an color image would have 3 dimensions.
 
"""
x_train.shape
x_test.shape

#  Reshape to include channel dimension (in this case, 1 channel)
x_train = x_train.reshape(60000, 28, 28, 1)
x_train.shape
x_test = x_test.reshape(10000,28,28,1)
x_test.shape

# Create the Model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



model.summary()


# Train the Model

# CHANGE NUMBER OF EPOCHS IF NECESSARY
# YOUR ACCURACY MAY ALSO BE LOWER THAN WHAT IS SHOWN HERE SINCE THIS WAS TRAINED ON GPU
model.fit(x_train,y_cat_train,epochs=2)

# Evaluate the Model

model.metrics_names
model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test)
y_cat_test.shape
y_cat_test[0]
predictions[0]
y_test
print(classification_report(y_test,predictions))



