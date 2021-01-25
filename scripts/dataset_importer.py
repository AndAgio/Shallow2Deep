import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import glob
import json
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.datasets import cifar10, mnist, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
# Import from files
import settings_parser


def get_generator_from_cifar(arguments, split_train=False, small=False):
    '''
    Function used to import data from cifar10 and create generators for train set,
    validation set (if needed) and test set.
    Params:
        - arguments: settings defined in the settings_parser file.
        - split_train: Boolean, True if splitting of train with validation is required.
        - small: Boolean, True if extracting a small subset of train and test
                 data is required. This is used to test evolution on small scale.
    Returns:
        - Tuple containing train, validation and test generators.
    '''
    # Get dataset from the keras library
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # If small, extract a small subset of train and test data
    if small:
        train_size = X_train.shape[0]
        X_train = X_train[:int(train_size*0.5),:,:,:]
        y_train = y_train[:int(train_size*0.5),:]
        X_test = X_test[:int(train_size*0.5),:,:,:]
        y_test = y_test[:int(train_size*0.5),:]
    # If split, get train shape and split train into train and validation
    if split_train:
        train_size = X_train.shape[0]
        X_val = X_train[int(0.8*train_size):, :, :, :]
        y_val = y_train[int(0.8*train_size):, :]
        X_train = X_train[:int(0.8*train_size), :, :, :]
        y_train = y_train[:int(0.8*train_size), :]
    # Preprocess Xs and labels to feed them to ImageDataGenerator
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(y_train, arguments.classes)
    Y_test = np_utils.to_categorical(y_test, arguments.classes)
    # Set ImageDataGenerator parameters for train
    train_datagen = ImageDataGenerator(rescale=1. / 255,  # rescale input image
                                       featurewise_center=False,  # set input mean to 0 over the dataset
                                       samplewise_center=False,  # set each sample mean to 0
                                       featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                       samplewise_std_normalization=False,  # divide each input by its std
                                       zca_whitening=False,  # apply ZCA whitening
                                       rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                       width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                                       height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                                       horizontal_flip=True,  # randomly flip images
                                       vertical_flip=False)  # randomly flip images)
    # Create train generator
    train_datagen.fit(X_train)
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=arguments.batch_size)
    # Create validation generator if needed
    if split_train:
        X_val = X_val.astype('float32')
        Y_val = np_utils.to_categorical(y_val, arguments.classes)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen.fit(X_val)
        val_generator = val_datagen.flow(X_val, Y_val, batch_size=arguments.batch_size)
        # Returns only train and validation
        return train_generator, val_generator
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen.fit(X_test)
    test_generator = test_datagen.flow(X_test, Y_test, batch_size=arguments.batch_size)
    # Returns only train and test
    return train_generator, test_generator


def get_generator_from_mnist(arguments, split_train=False, small=False):
    '''
    Function used to import data from cifar10 and create generators for train set,
    validation set (if needed) and test set.
    Params:
        - arguments: settings defined in the settings_parser file.
        - split_train: Boolean, True if splitting of train with validation is required.
        - small: Boolean, True if extracting a small subset of train and test
                 data is required. This is used to test evolution on small scale.
    Returns:
        - Tuple containing train, validation and test generators.
    '''
    # Get dataset from the keras library
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # If small, extract a small subset of train and test data
    if small:
        X_train = X_train[:500, :, :, np.newaxis]
        y_train = y_train[:500]
        X_test = X_test[:200, :, :, np.newaxis]
        y_test = y_test[:200]
    # If split, get train shape and split train into train and validation
    if split_train:
        train_size = X_train.shape[0]
        X_val = X_train[int(0.8*train_size):, :, :, np.newaxis]
        y_val = y_train[int(0.8*train_size):]
        X_train = X_train[:int(0.8*train_size), :, :, np.newaxis]
        y_train = y_train[:int(0.8*train_size)]
    # Preprocess Xs and labels to feed them to ImageDataGenerator
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(y_train, arguments.classes)
    Y_test = np_utils.to_categorical(y_test, arguments.classes)
    # Set ImageDataGenerator parameters for train
    train_datagen = ImageDataGenerator(rescale=1. / 255)  # rescale input image
    # Create train generator
    train_datagen.fit(X_train)
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=arguments.batch_size)
    # Create validation generator if needed
    if split_train:
        X_val = X_val.astype('float32')
        Y_val = np_utils.to_categorical(y_val, arguments.classes)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen.fit(X_val)
        val_generator = val_datagen.flow(X_val, Y_val, batch_size=arguments.batch_size)
        # Returns only train and validation
        return train_generator, val_generator
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen.fit(X_test)
    test_generator = test_datagen.flow(X_test, Y_test, batch_size=arguments.batch_size)
    # Returns only train and test
    return train_generator, test_generator


def get_generator_from_fashion_mnist(arguments, split_train=False, small=False):
    '''
    Function used to import data from cifar10 and create generators for train set,
    validation set (if needed) and test set.
    Params:
        - arguments: settings defined in the settings_parser file.
        - split_train: Boolean, True if splitting of train with validation is required.
        - small: Boolean, True if extracting a small subset of train and test
                 data is required. This is used to test evolution on small scale.
    Returns:
        - Tuple containing train, validation and test generators.
    '''
    # Get dataset from the keras library
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # If small, extract a small subset of train and test data
    if small:
        X_train = X_train[:500, :, :, np.newaxis]
        y_train = y_train[:500]
        X_test = X_test[:200, :, :, np.newaxis]
        y_test = y_test[:200]
    # If split, get train shape and split train into train and validation
    if split_train:
        train_size = X_train.shape[0]
        X_val = X_train[int(0.8*train_size):, :, :, np.newaxis]
        y_val = y_train[int(0.8*train_size):]
        X_train = X_train[:int(0.8*train_size), :, :, np.newaxis]
        y_train = y_train[:int(0.8*train_size)]
    # Preprocess Xs and labels to feed them to ImageDataGenerator
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(y_train, arguments.classes)
    Y_test = np_utils.to_categorical(y_test, arguments.classes)
    # Set ImageDataGenerator parameters for train
    train_datagen = ImageDataGenerator(rescale=1. / 255)  # rescale input image
    # Create train generator
    train_datagen.fit(X_train)
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=arguments.batch_size)
    # Create validation generator if needed
    if split_train:
        X_val = X_val.astype('float32')
        Y_val = np_utils.to_categorical(y_val, arguments.classes)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen.fit(X_val)
        val_generator = val_datagen.flow(X_val, Y_val, batch_size=arguments.batch_size)
        # Returns only train and validation
        return train_generator, val_generator
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen.fit(X_test)
    test_generator = test_datagen.flow(X_test, Y_test, batch_size=arguments.batch_size)
    # Returns only train and test
    return train_generator, test_generator
