import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class Unet:
    def __init__(self, verbose=False):
        inputs, outputs = self.get_layers()
        self.model = tf.keras.Model(inputs = inputs, outputs = outputs)
        if verbose:
            print(self.model.summary())

    def get_layers(self):
        Kernel = (3,3)
        pool_K = (2,2)
        NCHANNELSin = 1 # Number of input channels
        NCHANNELSout = 1 # Number of output channels
        inputs = Input((512,512,NCHANNELSin), name = 't')
        conv1 = Conv2D(64, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=pool_K)(conv1)
        conv2 = Conv2D(128, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#pool1
        conv2 = Conv2D(128, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=pool_K)(conv2)
        conv3 = Conv2D(256, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) #pool2
        conv3 = Conv2D(256, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=pool_K)(conv3)
        conv4 = Conv2D(512, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) # pool3
        conv4 = Conv2D(512, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=pool_K)(conv4)
        conv5 = Conv2D(1024, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_K)(conv5))
        merge6 = Concatenate(axis = -1)([conv4,up6])
        conv6 = Conv2D(512, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_K)(conv6))
        merge7 = Concatenate(axis = -1)([conv3,up7])
        conv7 = Conv2D(256, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_K)(conv7)) # conv7
        merge8 = Concatenate(axis = -1)([conv2,up8])
        conv8 = Conv2D(128, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_K)(conv8))
        merge9 = Concatenate(axis = -1)([conv1,up9])
        conv9 = Conv2D(64, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        outputs = Conv2D(2, Kernel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        outputs = Conv2D(NCHANNELSout, (1,1), padding = 'same', kernel_initializer = 'he_normal')(outputs)
        outputs = Activation('sigmoid', dtype='float32', name='segmentation')(outputs)
        return inputs, outputs
