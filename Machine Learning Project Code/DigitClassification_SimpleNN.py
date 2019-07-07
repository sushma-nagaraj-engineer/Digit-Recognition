import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from collections import Counter
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

class SimpleNN:
    def __init__(self):
        # loading data from MNIST into keras
        # features_train contains array of pixel values for the training data samples
        #labels_train contains the category to which each digit belongs
        #features_test contains array of pixel values for the test data samples
        #labels_test is the class to which the test data samples are classified into
        (self.features_train, self.labels_train), (self.features_test, self.labels_test) = mnist.load_data()
        #function call to transform data() to preprocess data
        self.transform_data()

    def transform_data(self):
        # flattening each image of size 28*28 into 784 pixel vector
        self.num_pixels = self.features_train.shape[1] * self.features_train.shape[2]
        
        #reshape() function is used to flatten image
        #In Keras, the layers used for two-dimensional convolutions expect pixel values with the dimensions [pixels][width][height]                 
        self.features_train = self.features_train.reshape(self.features_train.shape[0], 1, 28, 28).astype('float32')
        self.features_test =self.features_test.reshape(self.features_test.shape[0], 1, 28, 28).astype('float32')

        #Normalization is performed to give equal importance to all features   
        #pixel value range from 0-255 on the gray scale so we try to normalize it to values between 0 and 1
    
        self.features_train = self.features_train / 255
        self.features_test = self.features_test / 255
        
        #The output variable is an integer from 0 to 9
        #Transforming the vector of class integers into a binary matrix using to_categorical function
        #This is called Multi-class classification problem 
        self.labels_train = np_utils.to_categorical(self.labels_train)
        self.labels_test = np_utils.to_categorical(self.labels_test)
        self.num_classes = self.labels_test.shape[1] 

    def build_model(self):
        #Random number generator is initialized to a constant seed for reproducability of results
        seed = 7
        numpy.random.seed(seed)
        
        # create model
        #CNN is fit over 10 epochs with a batch size of 200.
        #Batch size is the number of input values we propogate through the network
        
        #Configuring the model for training
        model = Sequential()
        #Conv2D is a 2-dimentional convolutional network that creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
        model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        #Max pooling is a sample-based discretization process. It is used to reduce input dimensionality
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #Dropout is used to regularize data and reduce overfitting.
        model.add(Dropout(0.2))
        #Flattens the input. Does not affect the batch size.
        model.add(Flatten())
        #Dense implements the operation: output = activation(dot(input, kernel) + bias)
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(self.features_train, self.labels_train, validation_data=(self.features_test, self.labels_test), epochs=10, batch_size=200, verbose=2)
        # Final evaluation of the model
        scores = model.evaluate(self.features_test, self.labels_test, verbose=0)
        error=(100-scores[1]*100)
        return error

