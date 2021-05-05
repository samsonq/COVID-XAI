import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import preprocessing
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
import innvestigate
import innvestigate.utils as iutils


class ContrastiveLRP:
    """
    Contrastive Layer-wise Relevance Propagation
    """
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels
        self.label_dict = {j: i for i, j in enumerate(self.labels)}
        self.model_wo_sm = iutils.keras.graph.model_wo_softmax(self.model)
        o = keras.layers.Dense(3, use_bias=False, activation='linear', name='concept', trainable=False)(
            self.model_wo_sm.output)

        self.cmodel = keras.models.Model(inputs=model.input, outputs=o, name='concept_model')
        self.cmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    @staticmethod
    def rescale(img):
        return (img - img.min()) / (img.max() - img.min())

    @staticmethod
    def clrp(R, R_dual):
        return np.clip(R - R_dual, a_min=0, a_max=None)

    @staticmethod
    def conservation(H, Y, error=0.5):
        R = H.sum()
        c = np.abs((R - Y) / (Y + 1e-9)).max()
        print(c)
        return 0. <= c <= error

    @staticmethod
    def plot_images(imgs, suptitle=None, titles=None, save_path=None):
        n = len(imgs)
        if titles is None or len(titles) != n:
            titles = [""] * n

        fig, axes = plt.subplots(1, n)
        fig.set_figwidth(5 * n)
        #     fig.set_figheight(10)

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=16)

        for ax, img, t in zip(axes, imgs, titles):
            ax.set_title(t)

            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if save_path is not None:
            fig.savefig(save_path)

    @staticmethod
    def get_weights(output_size, idx, method="CLRP1"):
        W = np.zeros((output_size, 3))

        if method == "CLRP1":
            # concept weights
            W[idx, 0] = 1
            # not concept weights
            W[:, 1] = 1
        elif method == "CLRP2":
            # concept weights
            W[idx, 0] = 1
            # not concept weights
            W[idx, 1] = -1
        else:
            raise Exception("Unknown method")

        return W

    def analyze_concept(self, cmodel, idx, img_pp, rule="torch_lrp.sequential_preset_a", params={}):
        _, s = cmodel.layers[-2].output_shape

        W = self.get_weights(s, idx)
        cmodel.get_layer("concept").set_weights([W])

        canalyzer = innvestigate.create_analyzer(rule, cmodel, **params, neuron_selection_mode="index")

        R = canalyzer.analyze(img_pp, 0)
        R_dual = canalyzer.analyze(img_pp, 1)

        return R, R_dual

    def plot_CLRP(self, R, R_dual, title=None, save_path=None):
        imgs = [self.rescale(R[0]), self.rescale(R_dual[0]), self.rescale(self.clrp(R, R_dual)[0])]
        titles = ["R", "R_dual", "CLRP"]
        self.plot_images(imgs, title, titles, save_path=save_path)

    def C_LRP(self, img, label, save_path):
        """
        Computes contrastive-LRP for model on inputted image.
        :param img: input image path
        :param label: image label
        :param save_path: explanation save path
        :return: R, R_dual
        """
        def load_image(path):
            img_path = path
            img = preprocessing.image.load_img(img_path, target_size=(150, 150))
            x = preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return img, x

        _, img_pp = load_image(img)
        R, R_dual = self.analyze_concept(self.cmodel, self.label_dict[label], img_pp)
        self.plot_CLRP(R,
                       R_dual,
                       title="Type: {}".format(label),
                       save_path=save_path)
        return R, R_dual
