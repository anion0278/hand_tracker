
raise AssertionError("This version is not ready for use")

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
	@staticmethod
	def build(width, height, depth):

		filters = 32

		inputs = Input((width, height, depth))
		conv1 = Conv2D(filters, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(inputs)
		x = Conv2D(filters, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(conv1)
		x = BatchNormalization()(x)
		pool1 = MaxPooling2D(pool_size=(2, 2))(x)
		conv2 = Conv2D(filters * 2, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(pool1)
		x = Conv2D(filters * 2, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(conv2)
		x = BatchNormalization()(x)
		pool2 = MaxPooling2D(pool_size=(2, 2))(x)
		conv3 = Conv2D(filters *4, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(pool2)
		x = Conv2D(filters * 4, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x)
		pool3 = MaxPooling2D(pool_size=(2, 2))(x)
		conv4 = Conv2D(filters * 8, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(pool3)
		x = Conv2D(filters * 8, 3, activation = 'relu', use_bias=False, padding = 'same', kernel_initializer = 'he_normal')(conv4)
		x = BatchNormalization()(x)
		drop4 = Dropout(0.5)(x)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(filters * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4,up6], axis = 3)
		conv6 = Conv2D(filters * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(filters * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(filters * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(filters * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(filters * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(filters * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(input = inputs, output = conv10)

		return model

