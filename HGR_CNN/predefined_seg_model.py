from segmentation_models import Unet
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import os
import model_wrapper as mw
import config

def build(input_size):
    base_model = Unet(backbone_name='resnet34', encoder_weights='imagenet')

    inp = Input(shape=(None, None, 1))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    return Model(inp, out, name=base_model.name)
