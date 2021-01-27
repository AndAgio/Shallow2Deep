import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import glob
import json
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
# Import from files
import settings_parser

def get_standard_callbacks(args, model_path, initializer=False, scheduler=True,
                           plateau=False, printer=False, stopper=False,
                           nan=True, board=False, storer=True):
    '''
    Function that sets up standard callbacks used during the training of a model.
    Params:
        - args: settings defined in the settings_parser file.
        - model_path: string defining where the best model will be stored in a h5 file.
        - initializer: True if learning rate initializer is needed.
        - scheduler: True if learning rate scheduler is needed.
        - plateau: True if learning rate reduce on plateau is needed.
        - printer: True if learning rate printer is needed.
        - stopper: True if early stopper is needed.
        - nan: True if stop when loss is NaN is needed.
        - board: True if tensorboard is needed.
        - storer: True if model checkpoint is needed.
    Returns:
        - List: list of callbacks to use during training.
    '''
    # Define callbacks
    my_callbacks = []
    if initializer:
        lr_init_callback = CustomLearningRateInitializer(lr_start=args.lr_start)
        my_callbacks.append(lr_init_callback)
    if scheduler:
        lr_sched_callback = CustomLearningRateScheduler(epochs=args.n_epochs,
                                                        lr_start=args.lr_start,
                                                        lr_decay_epochs=args.lr_decay_epochs,
                                                        lr_decay_factor=args.lr_decay_factor)
        my_callbacks.append(lr_sched_callback)
    if plateau:
        plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                factor=0.1,
                                                                patience=5,
                                                                min_lr=1e-6)
        my_callbacks.append(plateau_callback)
    if printer:
        lr_printer_callback = CustomLearningRatePrinter()
        my_callbacks.append(lr_printer_callback)
    if stopper:
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                               min_delta=0.01,
                                                               patience=10)
        my_callbacks.append(early_stop_callback)
    if nan:
        terminate_nan_callback = tf.keras.callbacks.TerminateOnNaN()
        my_callbacks.append(terminate_nan_callback)
    if board:
        # For tensorboard check if folder is available
        tensorboard_folder = os.path.join(args.log_folder, args.tensorboard_folder)
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_folder,
                                                     update_freq='epoch')
        my_callbacks.append(tb_callback)
    if storer:
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                           monitor='val_accuracy',
                                                           verbose=0,
                                                           save_best_only=True,
                                                           save_weights_only=False)
        my_callbacks.append(save_callback)

    return my_callbacks


def progressive_learning_rate(steps_per_epoch, epochs=50, lr_start=0.1, lr_decay_epochs=10, lr_decay_factor=10):
    print('Steps per epoch: {}'.format(steps_per_epoch))
    boundaries = [i*steps_per_epoch for i in range(lr_decay_epochs,epochs,lr_decay_epochs)]
    print('Boundaries: {}'.format(boundaries))
    values = [lr_start/pow(lr_decay_factor,i) for i in range(len(boundaries)+1)]
    print('Values: {}'.format(values))
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    return learning_rate_fn


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """
    Learning rate scheduler which sets the learning rate according to schedule.
    Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
    """
    def __init__(self, epochs=50, lr_start=0.1, lr_decay_epochs=10, lr_decay_factor=10):
        super(CustomLearningRateScheduler, self).__init__()
        self.epochs = epochs
        self.lr_start = lr_start
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_factor = lr_decay_factor

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        # print('Epoch {}: Learning rate is {:.5f}'.format(epoch, scheduled_lr))

    def schedule(self, epoch):
        # print('Intervals number: {}'.format(self.epochs // self.lr_decay_epochs))
        # If decay period is bigger than total number of epochs, set learning rate to its start value
        if self.lr_decay_epochs > self.epochs:
            learning_rate = self.lr_start
        for i in range(self.epochs // self.lr_decay_epochs + 1):
            interval = [i * self.lr_decay_epochs, (i + 1) * self.lr_decay_epochs - 1]
            #print('epoch: {} & interval: {}'.format(epoch, interval))
            if interval[0] <= epoch <= interval[1]:
                learning_rate = self.lr_start / np.power(self.lr_decay_factor, i)
        #print('Epoch: {} -> Learning rate: {:.5f}'.format(epoch, learning_rate))
        return learning_rate


class CustomLearningRatePrinter(keras.callbacks.Callback):
    """
    Learning rate printer which prints the learning rate at the beginning of each epoch.
    """
    def __init__(self):
        super(CustomLearningRatePrinter, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Print learning rate at the beginning of each epoch
        print('Epoch {}: Learning rate is {:.6f}'.format(epoch+1, keras.backend.eval(self.model.optimizer.lr)))


class CustomLearningRateInitializer(keras.callbacks.Callback):
    """
    Learning rate initializer which sets the learning rate at the beginning of training.
    """
    def __init__(self, lr_start=0.1):
        super(CustomLearningRateInitializer, self).__init__()
        self.lr_start = lr_start

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Initialize learning rate at the beginning of training
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr_start)
