from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import tensorflow as tf
import numpy as np
import os
    
class DatasetManager:   
    def __init__(self,config):
        self.config = config
        self.img_dataset_size = config.img_dataset_size

    def get_autoencoder_datagens(self, jit=True):
        # jit -> load images in batches from folders, not all together
        if jit:
            return self.__get_jit_autoencoder_datagens(self.config.dataset_dir_path, self.config.imgs_dir, self.config.masks_dir,self.config.val_imgs_dir,self.config.val_masks_dir,self.config.batch_size)
        else:
            raise ModuleNotFoundError("Not implemented yet!")

    def __get_jit_autoencoder_datagens(self, dataset_path, imgs_dir, masks_dir,val_imgs_dir,val_masks_dir,batch_size):
        augmentations = dict(#featurewise_center=True,
                            #featurewise_std_normalization=True,
                            #dtype = datatype
                            rescale=1. / 255.0,
                            horizontal_flip = True,
                            vertical_flip = True)

        seed = 1
        image_generator = self.__get_datagen(augmentations, seed, dataset_path, imgs_dir, batch_size)
        image_val_generator = self.__get_validation_datagen(augmentations,seed,dataset_path,val_imgs_dir,batch_size)

        #image_dataset=tf.data.Dataset.from_generator(lambda:image_generator,output_signature=(tf.TensorSpec(shape=(*reversed(self.img_dataset_size),1),dtype=tf.float32)))
        #image_val_dataset=tf.data.Dataset.from_generator(lambda:image_val_generator,output_signature=(tf.TensorSpec(shape=(*reversed(self.img_dataset_size),1),dtype=tf.float32)))
        #TODO add preprocessing_function = binarization_norm
        mask_generator = self.__get_datagen(augmentations, seed, dataset_path, masks_dir, batch_size)
        mask_val_generator = self.__get_validation_datagen(augmentations,seed,dataset_path,val_masks_dir,batch_size)

        #mask_dataset=tf.data.Dataset.from_generator(lambda:mask_generator,output_signature=(tf.TensorSpec(shape=(*reversed(self.img_dataset_size),1),dtype=tf.float32)))
        #mask_val_dataset=tf.data.Dataset.from_generator(lambda:mask_val_generator,output_signature=(tf.TensorSpec(shape=(*reversed(self.img_dataset_size),1),dtype=tf.float32)))

        train_steps = np.floor(len(os.listdir(os.path.join(dataset_path, imgs_dir))) / batch_size)
        val_train_steps = np.floor(len(os.listdir(os.path.join(dataset_path, val_imgs_dir))) / batch_size)

        train_gen = (pair for pair in zip(image_generator, mask_generator))
        val_gen = (pair for pair in zip(image_val_generator, mask_val_generator))

        #train_gen = (pair for pair in tf.data.Dataset.zip((image_dataset, mask_dataset)))
        #val_gen = (pair for pair in tf.data.Dataset.zip((image_val_dataset, mask_val_dataset)))
        # TODO separate train from val:
        # https://github.com/keras-team/keras/issues/5862
        return train_gen, train_steps, val_gen, val_train_steps

    def __get_datagen(self, augmentations, seed, dataset_dir, class_dir, batch_size):
        datagen = ImageDataGenerator(**augmentations,) 
        #preprocessing_function = binarization_norm
        generator = datagen.flow_from_directory(dataset_dir,
                                                        class_mode=None,
                                                        classes=[class_dir],
                                                        color_mode="grayscale",
                                                        target_size=reversed(self.img_dataset_size), # h, w !
                                                        seed=seed, batch_size = batch_size)
        generator.next()
        return generator

    def __get_validation_datagen(self, augmentations, seed, dataset_dir, class_dir, batch_size):
        datagen = ImageDataGenerator(**augmentations,) 
        #preprocessing_function = binarization_norm
        generator = datagen.flow_from_directory(dataset_dir,
                                                        class_mode=None,
                                                        classes=[class_dir],
                                                        color_mode="grayscale",
                                                        target_size=reversed(self.img_dataset_size), # h, w !
                                                        seed=seed, batch_size = batch_size)
        generator.next()
        return generator


    def get_eval_datagens(self):
        dataset_path = self.config.camera_image_dir
        imgs_dir = self.config.camera_depth_dir
        masks_dir = self.config.camera_label_dir
        batch_size = 50
        augmentations = dict(#featurewise_center=True,
                            #featurewise_std_normalization=True,
                            #dtype = datatype
                            rescale=1. / 255.0,
                            horizontal_flip = True,
                            vertical_flip = True)
        seed = 1
        image_generator = self.__get_eval_datagen(augmentations, seed, dataset_path, imgs_dir, batch_size)
        mask_generator = self.__get_eval_datagen(augmentations, seed, dataset_path, masks_dir, batch_size)
        eval_gen = (pair for pair in zip(image_generator, mask_generator))
        eval_steps = np.floor(len(os.listdir(os.path.join(dataset_path, imgs_dir))) / batch_size)

        return eval_gen,eval_steps 
    def __get_eval_datagen(self, augmentations, seed, dataset_dir, class_dir, batch_size):
        datagen = ImageDataGenerator(**augmentations,)
        generator = datagen.flow_from_directory(dataset_dir,
                                                        class_mode=None,
                                                        classes=[class_dir],
                                                        color_mode="grayscale",
                                                        target_size=reversed(self.img_dataset_size), # h, w !
                                                        seed=seed, batch_size = batch_size)
        generator.next()
        return generator


