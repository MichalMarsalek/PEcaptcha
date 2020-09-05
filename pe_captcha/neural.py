from os import listdir
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.experimental.preprocessing import Rescaling
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import pickle

def model0(training_dir):
    #This model has 98.8% validation accuracy on individual characters (those that are correctly deparated)
    #This would suggest 94% accuracy on 5-tuples. Most of the errors (15/22) is cost by wrong splitting and
    #accuracy on real CAPTCHAs is 90%.
    data = []
    labels = []
    for i in range(10):
        dir = join(training_dir, str(i))
        for file in listdir(dir):
            image = cv.imread(f"{dir}\\{file}")
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)
            if image.shape == (24,24,1):
                data.append(image)
                labels.append(i)

    data = np.array(data)
    labels = np.array(labels)
    print("data size", len(data))

    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    model = Sequential()

    model.add(Rescaling(1.0/255))
    # First convolutional layer with max pooling
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(24,24, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # Second convolutional layer with max pooling
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))

    # Output layer with 10 nodes (one for each possible letter/number we predict)
    model.add(Dense(10, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the neural network
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=15, verbose=1)

    # Return model
    return model

model_t = model0