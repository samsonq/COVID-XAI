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
from sklearn.utils import class_weight
from config import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


# TODO: refactor config.py into yaml file so no need to import values?
class PneumoniaModel:
    """

    """
    def __init__(self, architecture="VGG16"):
        print("GPU in use:", tf.test.is_gpu_available())
        device_lib.list_local_devices()

        self.architecture = architecture
        self.model = self.__build_model()

    def __str__(self):
        return self.model.summary()

    def __build_model(self):
        K.clear_session()
        dim = 1 if GRAYSCALE else 3
        if self.architecture == "VGG16":
            base_model = VGG16(weights='./models/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                               include_top=False,
                               input_shape=(IMG_SIZE, IMG_SIZE, dim))
        elif self.architecture == "ResNet18":
            base_model = Sequential()
            base_model.add(ResNet50(include_top=False,
                                    pooling='avg',
                                    input_shape=(IMG_SIZE, IMG_SIZE, dim),
                                    weights='./models/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(len(LABELS), activation='softmax', name="concept")(x)
        model = Model(inputs=base_model.input, outputs=x)
        for layer in model.layers[0:19]:
            layer.trainable = False
        return model

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

    @staticmethod
    def __callbacks():
        callbacks = []
        return callbacks

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
        ohe_y_train = keras.utils.to_categorical(y_train, len(LABELS))
        ohe_y_val = keras.utils.to_categorical(y_valid, len(LABELS))

        w_array = np.ones((2, 2))
        w_array[1, 0] = 30  # penalizing false negative
        w_array[0, 1] = 1  # penalizing false positive

        spec_loss = lambda y_true, y_pred: self.w_categorical_crossentropy(y_true, y_pred, weights=w_array)

        y_labels = np.argmax(ohe_y_train, axis=1)
        classweight = class_weight.compute_class_weight("balanced", np.unique(y_labels), y_labels)

        optimizer = Adam(lr=0.0001)

        self.model.compile(loss="categorical_crossentropy",  # loss=spec_loss,
                           optimizer=optimizer,
                           metrics=["accuracy"])

        history = self.model.fit(x=X_train, y=ohe_y_train,
                                 class_weight=classweight,
                                 validation_data=(X_valid, ohe_y_val),
                                 shuffle=True,
                                 callbacks=self.__callbacks(),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1)
        return history
