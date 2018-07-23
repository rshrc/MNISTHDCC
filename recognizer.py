# Plotting and Visualizing the MNIST Dataset

from keras.datasets import mnist
import matplotlib.pyplot as plt

# This will download the dataset for us
# dataset = mnist.load_data() 

(X_train, y_train), (X_test, y_test) = dataset

# Plotting four images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# show th plot
plt.show()

import numpy as np
from keras.models import Sequential
from keras.layers import (
        Dense,
        Dropout,
)
from keras.utils import np_utils

np.random.seed(0) # Setting the random seed

"""
We will reduce the images down into a vector of pixels
In this case 28x28 sized images will be 784 pixel input values
"""

# Memory requirements will be reduced by making the precision values to be 
# float32

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

"""
This is a multi class classification problem,
Using one hot encoding of the class values,
transforming the vector of class integers into a binary matrix
"""

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Creating the neural network
def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    
    # Compiling model together
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

"""
The model is a simple neural network with one
hidden layer with the same number of inputs that is 784.
A ReLU (Rectifier Linear Unit) is used for the neurons
in the hidden layer

A softmax activation function is used on the output layer to turn
the outputs into probability like values
and allows one class of the 10 to be selected as the model's output
prediction. Logarithimic loss is used as the loss function 
(called categorical_crossentropy in Keras) and 
the efficient ADAM gradient descent algorithm is used for the learning
of the weights in the neurons
"""

# build the model
model = baseline_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error is {}".format(100-scores[1]*100))

