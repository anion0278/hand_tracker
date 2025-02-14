import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np

def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=size, padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x) # Leaky ReLU should be used as a layer, not as activation
    x = Conv2D(filters=nfilters, kernel_size=size, padding=padding, kernel_initializer=initializer)(x)
    # TODO try network in network approach x = Conv2D(filters=int(nfilters/2), kernel_size=1, padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = UpSampling2D()(tensor)
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), padding=padding)(y)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y

def build(input_size, filters=16):
# encoder
    # input 240 x 320 x 1
    input_layer = Input(shape=(input_size[1], input_size[0], 1), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)

# decoder
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)

# output
    output_layer = Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    #for multiclass segmentation: 
    #output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    #output_layer = BatchNormalization()(output_layer)
    #output_layer = Activation('softmax')(output_layer)
    # + loss="sparse_categorical_crossentropy"

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    model.summary()
    # TODO show model graph tf.keras.utils.plot_model
    

    return model