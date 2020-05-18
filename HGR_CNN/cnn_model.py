
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np

import matplotlib.pyplot as plt  
import datetime  
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class CnnModel:
    def __init__(self, conv_filters, learning_rate, image_sqr_size, existing_model_name=None):
        if (existing_model_name == None):
            self.model = self.__create_model(conv_filters, learning_rate, image_sqr_size)
        else:
            self.model = tf.keras.models.load_model(existing_model_name)

    # def  TODO tf.test.gpu_device_name() -> will return /device:GPU:0 is is running on GPU

    def train(self, X_dataset, y_dataset, nb_epoch, batch_size, logs_path, test_data_ratio):

        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=test_data_ratio, random_state=42)

        print("Train data length: %s" % len(X_train))
        print("Test data length: %s" % len(X_test))

        train_datagen = ImageDataGenerator(rescale=1. / 255) # change 0..255 to 0..1

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # TODO make multistage training -> divide training into phases and save network after each phase
        #validation_split=0.20 - instead of splitting, but the data has to be shuffled beforehand!
        try:
            tensorboard = TensorBoard(log_dir=logs_path)

            cp_callback = ModelCheckpoint(filepath=os.path.join("models","checkpoints","checkpoint_model.h5"),  # TODO fix - pass as arg
                                            save_weights_only=False, verbose=1,
                                            monitor='val_loss',
                                            period=int(nb_epoch/10),# every X epochs
                                            #save_best_only=True
                                            ) 

            reduce_LR_callback = ReduceLROnPlateau(monitor='val_loss', verbose=1, # reduces LR during stagnation 
                                factor=0.1, 
                                patience=2, # number of epochs to wait
                                mode='auto', 
                                min_delta=0.001,  # min change
                                cooldown=0, 
                                min_lr=0.000001)  # min learning rate

            early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=6, verbose=1, mode='auto')

            # LearningRateScheduler - can be used to set learning rate at any epoch

            # just csv logger
            #CSVLogger(filename, separator=',', append=False)

            # for custom callbacks
            #LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)

            result = self.model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                #steps_per_epoch=40, defines how many batches will be recieved
                                # since dataaugmentation runs only once at the begining of the epoch, 
                                # it makes sence to only run x_train.size/batch_size epochs  
                                #if not defined -> will train exactly x_train.size/batch_size
                                epochs=nb_epoch,
                                validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
                                #validation_steps=20, # if not defined -> fill run
                                #all valid data once
                                verbose=1,
                                callbacks=[tensorboard, cp_callback, reduce_LR_callback, early_stopping_callback]) #LearningRateCallback(),
            # TODO use initial_epoch for training continuation

        except KeyboardInterrupt:
            #self.model.save("interrupted_model.h5")
            pass # will just continue on saving 

    def predict_single_image(self, X_img, y_expected):
        test_datagen = ImageDataGenerator(rescale=1. / 255) # TODO Exctract as common method for learning and prediction processes
        testData = test_datagen.flow(np.array([X_img]), np.array([y_expected]), batch_size=1)
        prediction = self.model.predict(testData)
        return np.squeeze(prediction)

    def save(self, model_name):
        self.model.save(model_name)
        print("Model saved: %s" % model_name) 

    def __create_model(self, conv_filters, learning_rate, image_size):
        activation_function = LeakyReLU(alpha=0.1)
        pooling_kernel = 2
        input_shape = (image_size[1], image_size[0], 1)

        model = Sequential()
        model.add(Input(shape=input_shape))
    
        self.add_conv_layer(model,conv_filters, 5, pooling_kernel, activation_function, name="ConvInput")
        for i in range(2, 4):
            # if kernel size is 1 number - it is defined for all axes 3 = (3,3)
            self.add_conv_layer(model, conv_filters * i, 5, pooling_kernel, activation_function, name=f"Conv{i}")

        model.add(Flatten())
        # TODO add more Batch norm here??? should not be relevant
        model.add(Dense(image_size[0] * image_size[1] / 10)) # just to fit model into GPU
        model.add(Dropout(0.5)) # Fraction of the input units to drop(!). droupout should be after normalization. 

        model.add(Dense(5, activation="linear")) 

        #Nadam(learning_rate=learning_rate) #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        #Adadelta(learning_rate=learning_rate)
        opt = Adam(learning_rate=learning_rate)

        model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['mse']) # TODO - move to "train" method
        model.summary()
        return model

    def add_conv_layer(self, model, conv_filters, conv_kernel, pooling_kernel, activation_function, name):
        # Kernel size should be odd number
        # should be a method of ModelBuilder
        model.add(Conv2D(conv_filters, kernel_size=conv_kernel, use_bias=False, padding='same', name=name, kernel_initializer = 'he_uniform')) 
        # He initialization is recommended for ReLU
        model.add(Activation(activation_function))
        model.add(BatchNormalization()) # Normalization after Conv layer (ResNet Keras)
        # axis should be set to Channels (w, h, ch), default is -1 (the last dimension)
        #TODO check performance when BN is before/after Activation - both are correct sequences
        # Maybe BN before ReLU allows better performance after optimalization (TensorRT)

        model.add(Conv2D(int(conv_filters / 2), kernel_size=1, use_bias=False, padding='same', name=name +"-1x1", kernel_initializer = 'he_uniform')) 
        # he_uniform is for ReLU
        # 1x1 conv with 1/2 filters for dimensionality reduction
        # TODO Check 1/4 reduction
        model.add(Activation(activation_function))
        model.add(BatchNormalization()) 
        model.add(MaxPooling2D(pool_size=pooling_kernel, padding='same'))
        # No dropout for conv layers, because Dropout should not be used before ANY Batch Norm