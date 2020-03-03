
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt    


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout, Activation, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split

class CnnModel:
    def __init__(self, conv_filters, learning_rate, image_sqr_size, existing_model_name=None):
        if (existing_model_name == None):
            self.model = self.__create_model(conv_filters, learning_rate, image_sqr_size)
        else:
            self.model = tf.keras.models.load_model(existing_model_name)
            self.setup_graph()

    def train(self, X_dataset, y_dataset, nb_epoch, batch_size, logs_path, test_data_ratio):

        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=test_data_ratio, random_state=42)

        print("Train data length: %s" % len(X_train))
        print("Test data length: %s" % len(X_test))

        train_datagen = ImageDataGenerator(rescale=1. / 255) # change 0..255 to 0..1

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        tensorboard = TensorBoard(log_dir=logs_path)
        # TODO make multistage training -> divide training into phases and save network after each phase
        #validation_split=0.20 - instead of splitting, but the data has to be
        #shuffled beforehand!
        result = self.model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                            #steps_per_epoch=40, # if not defined -> will train
                            #use exactly x_train.size/batch_size
                            epochs=nb_epoch,
                            validation_data=test_datagen.flow(X_test, y_test, batch_size=batch_size),
                            #validation_steps=20, # if not defined -> fill run
                            #all valid data once
                            verbose=1,
                            callbacks=[tensorboard])
        #self.show_history(result)
        self.setup_graph()

    #def show_history(self, result):
    #    # plot training curve for R^2 (beware of scale, starts very low negative)
    #    plt.plot(result.history['val_r_square'])
    #    plt.plot(result.history['r_square'])
    #    plt.title('model R^2')
    #    plt.ylabel('R^2')
    #    plt.xlabel('epoch')
    #    plt.legend(['train', 'test'], loc='upper left')
    #    plt.show()
           
    #    # plot training curve for rmse
    #    plt.plot(result.history['rmse'])
    #    plt.plot(result.history['val_rmse'])
    #    plt.title('rmse')
    #    plt.ylabel('rmse')
    #    plt.xlabel('epoch')
    #    plt.legend(['train', 'test'], loc='upper left')
    #    plt.show()

    def predict_single_image(self, X_img, y_expected):
        test_datagen = ImageDataGenerator(rescale=1. / 255) # TODO Exctract as common method for learning and prediction processes
        testData = test_datagen.flow(np.array([X_img]), np.array([y_expected]), batch_size=1)
        #with session.as_default():
            #with graph.as_default():
        prediction = self.model.predict(testData)
        return np.squeeze(prediction)

    def setup_graph(self):
        self.model._make_predict_function()
        #global session
        #session = tf.keras.backend.get_session()
        #global graph
        #graph = tf.get_default_graph()    

    def save(self, model_name):
        self.model.save(model_name)
        print("Model saved: %s" % model_name) 

    def rmse(y_true, y_pred):
        import tf.keras.backend
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def __create_model(self, conv_filters, learning_rate, image_size):
        conv_kernel = (3, 3)
        pooling_kernel = (2, 2)
        activation_function = "relu"
        input_shape = (image_size[1], image_size[0], 1)

        model = Sequential()

        model.add(Conv2D(conv_filters, kernel_size=conv_kernel, use_bias=False, input_shape=input_shape))
        model.add(BatchNormalization()) # axis should be set to Channels (w, h, ch), default is -1 (the last dimension)
        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(Activation(activation_function))
        # No dropout for conv layers, because Dropout should not be used before ANY Batch Norm

        model.add(Conv2D(conv_filters, kernel_size=conv_kernel, use_bias=False)) # TODO add padding='SAME'
        model.add(BatchNormalization()) # Normalization after Conv layer, but can be after ReLU (compare?)
        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(Activation(activation_function))

        model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, use_bias=False))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(Activation(activation_function))

        model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, use_bias=False))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(Activation(activation_function))

        model.add(Conv2D(conv_filters * 2, kernel_size=conv_kernel, use_bias=False)) 
        model.add(BatchNormalization()) 
        model.add(MaxPooling2D(pool_size=pooling_kernel))
        model.add(Activation(activation_function))

        model.add(Flatten())
        model.add(Dense(image_size[0] * image_size[1] * 2))
        model.add(Dropout(0.5)) # Fraction of the input units to drop(!). droupout should be after normalization. 

        model.add(Dense(5, activation="linear")) 

        opt = Adam(learning_rate=learning_rate)

        model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['mse'])
        return model
