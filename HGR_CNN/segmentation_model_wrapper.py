from segmentation_models import Unet
from keras.layers import Input, Conv2D
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import os

import config as c

def create_model(model_path, config):
    if not config.use_gpu:
        disable_gpu()


    base_model = Unet(backbone_name='resnet34', encoder_weights='imagenet')

    inp = Input(shape=(None, None, 1))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)

    return SModelWrapper(model, config)

def dice_coef(y_true, y_pred):
    # should be used due to unbalanced labels
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def check_gpu():
    if tf.test.is_gpu_available(): #TF2.1 tf.config.list_physical_devices("GPU"):
        c.msg("GPU is available")
    else:
        c.msg("GPU is NOT available")

def disable_gpu():
    try: # TODO - fix exception when a new model is build before this param is set
        tf.config.experimental.set_visible_devices([], "GPU")
    except:
        pass

class SModelWrapper():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        if not config.use_gpu:
            disable_gpu()

    def save_model_graph_img(self):
        img_name = self.config.version_name+".png"
        #tf.keras.utils.plot_model(self.model, to_file = img_name, expand_nested=True, show_shapes=True)  #rankdir = "LR",
        return img_name

    def train(self, train_data_gen, train_steps, val_data_gen, val_steps):
        #tensorboard = TensorBoard(log_dir=self.config.logs_dir)

        # TODO monitor "loss" -> change to "val_loss" if val_data_gen is available and val_steps > 1

        cp_callback = ModelCheckpoint(filepath=os.path.join(self.config.models_dir,"checkpoints","latest_checkpoint.h5"), 
                                        save_weights_only=False, verbose=1,
                                        monitor="loss",
                                        #period=1,# every X epochs
                                        save_freq="epoch",
                                        save_best_only=True) 

        reduce_LR_callback = ReduceLROnPlateau(monitor="loss", verbose=1, # reduces LR during stagnation
                            factor=0.01, 
                            patience=1, # number of epochs to wait
                            mode="auto", 
                            min_delta=0.05,  # min change
                            cooldown=0, 
                            min_lr= self.model.optimizer.lr / 500)  # min learning rate

        #early_stopping_callback = EarlyStopping(monitor="loss", min_delta=0.01, patience=5, verbose=1, mode="auto")

        history = self.model.fit(train_data_gen, validation_data=val_data_gen, 
                                  epochs = self.config.epochs_count, steps_per_epoch = train_steps, validation_steps = val_steps,
                                  max_queue_size=50,                # maximum size for the generator queue
                                  workers=8,                        # maximum number of processes 
                                  use_multiprocessing=True,         # use threading (maybe does not work on Win)
                                  #shuffle=True does not work with generators that do not implement keras.utils.Sequence
                                  callbacks = [cp_callback, reduce_LR_callback])  
                                  #callbacks = [tensorboard, cp_callback, reduce_LR_callback])

    def recompile(self):
        self.model.compile(loss=dice_loss,optimizer=Adam(learning_rate=self.config.learning_rate), metrics=["binary_crossentropy",tf.keras.metrics.MeanIoU(num_classes=2)])

    def save(self, model_path):
        self.model.save(model_path)
        c.msg(f"Model saved to: {model_path}")