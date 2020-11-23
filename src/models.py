import os
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras import preprocessing
from tensorflow.python.client import device_lib
from config import LABELS, IMG_SIZE, GRAYSCALE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class PneumoniaModel:
    """

    """
    def __init__(self):
        print("GPU in use:", tf.test.is_gpu_available())
        device_lib.list_local_devices()

        K.clear_session()

        dim = 1 if GRAYSCALE else 3
        base_model = VGG16(weights='./weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, dim))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(len(LABELS), activation='softmax', name="concept")(x)

        self.model = Model(inputs=base_model.input, outputs=x)

    def __str__(self):
        return self.model.summary()

    @staticmethod
    def w_categorical_crossentropy(y_true, y_pred, weights):
        """

        :param y_true:
        :param y_pred:
        :param weights:
        :return:
        """
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
