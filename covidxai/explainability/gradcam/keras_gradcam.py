import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras
from keras.models import load_model
import cv2
import vis
from vis.visualization import visualize_cam
import warnings
warnings.filterwarnings("ignore")


class GradCAM:
    """

    """
    def __init__(self, model):
        """
        Specify model to perform Grad-CAM with.
        :param model: model object or filepath
        """
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.adjusted_model = self.model
        self.gradients = None

    def fit_grad_cam(self, img, last_layer, penultimate_layer,
                     preprocessing=None, show=True, save_path=None):
        """
        Compute Grad-CAM output with inputted image.
        :param img: image array or filepath
        :param last_layer: name of model's last layer
        :param penultimate_layer: name of model's last convolution layer
        :param preprocessing: function to preprocess image
        :param show: show grad-cam output
        :param save_path: save path for grad-cam output
        :return: model gradients
        """
        if isinstance(img, str):
            try:
                img = cv2.imread(img)
            except Exception as e:
                print("Invalid file path.")

        img = preprocessing(img)
        y_pred = self.model.predict(img.reshape(-1, img.shape[0], img.shape[1], img.shape[2]))
        class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
        class_idx = class_idxs_sorted[0]

        layer_idx = vis.utils.find_layer_idx(self.model, last_layer)
        self.adjusted_model = self.model

        # Swap softmax with linear
        self.adjusted_model.layers[layer_idx].activation = keras.activations.linear
        self.adjusted_model = vis.utils.apply_modifications(self.adjusted_model)

        penultimate_layer_idx = vis.utils.find_layer_idx(self.adjusted_model, penultimate_layer)
        seed_input = img
        self.gradients = visualize_cam(self.adjusted_model, layer_idx, class_idx, seed_input,
                                       penultimate_layer_idx=penultimate_layer_idx,
                                       backprop_modifier=None,
                                       grad_modifier=None)

        if show:
            self.show_grad_cam(img, self.gradients)
        if save_path:
            plt.imsave(save_path, self.gradients)
        return self.gradients

    @staticmethod
    def show_grad_cam(img, gradients):
        """
        Display model gradients with image.
        :param img: image to visualize
        :param gradients: model gradients
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(img)
        axes[1].imshow(img)
        i = axes[1].imshow(gradients, cmap="jet", alpha=0.8)
        fig.colorbar(i)
