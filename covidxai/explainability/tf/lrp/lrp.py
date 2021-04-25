import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import keras
import tensorflow as tf
import innvestigate
import innvestigate.utils as iutils
from matplotlib import pyplot as plt
import lrp_utils as utils


class LayerwiseRelevancePropagation:
    """
    Layer-wise Relevance Propagation
    """
    def __init__(self, model, images, batch_size, label_to_class_name,
                 input_postprocessing=None):
        """

        :param model: trained model
        :param images: image training set
        :param batch_size: batch size to fit analyzer
        :param label_to_class_name: dictionary of label names
        :param input_postprocessing: function to revert image preprocessing
        """
        self.model = model
        self.label_to_class_name = label_to_class_name
        self.methods = [
            # NAME, OPT.PARAMS, POSTPROC FUNC, TITLE
            ("input", {}, input_postprocessing, "Input"),

            # Signal
            ("deconvnet", {}, utils.bk_proj, "Deconvnet"),
            ("lrp.z", {}, utils.heatmap, "LRP-Z"),
            ("lrp.epsilon", {"epsilon": 1}, utils.heatmap, "LRP-Epsilon")
        ]

        self.input_postprocessing = input_postprocessing

        # Create model without trailing softmax
        self.model_wo_softmax = iutils.keras.graph.model_wo_softmax(self.model)
        # Create analyzers.
        self.analyzers = []
        for method in self.methods:
            analyzer = innvestigate.create_analyzer(method[0],  # analysis method identifier
                                                    self.model_wo_softmax,  # model without softmax output
                                                    **method[1])  # optional analysis parameters

            # Some analyzers require training.
            analyzer.fit(images, batch_size=batch_size, verbose=1)
            self.analyzers.append(analyzer)

    def lrp_output(self, img, label, method="z", save_path=None):
        """

        :param img: input image path
        :param label: image label
        :param method: LRP method, 'z' or 'epsilon'
        :param save_path: explanation save path
        :return: predicted, actual label, logit, probability
        """
        assert method in ["z", "epsilon"], "Unknown LRP method."
        img = utils.load_image(img, 150)
        analysis = np.zeros([1, len(self.analyzers), img.shape[1], img.shape[2], len(self.methods)-1])
        text = []

        img = img[None, :, :, :]

        # Predict final activations, probabilities, and label.
        presm = self.model_wo_softmax.predict_on_batch(img)[0]
        prob = self.model.predict_on_batch(img)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(("%s" % self.label_to_class_name[label],  # ground truth label
                     "%.2f" % presm.max(),  # pre-softmax logits
                     "%.2f" % prob.max(),  # probabilistic softmax output
                     "%s" % self.label_to_class_name[y_hat]  # predicted label
                     ))

        for aidx, analyzer in enumerate(self.analyzers):
            # Analyze.
            a = analyzer.analyze(img)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = utils.postprocess(a)
            # Apply analysis postprocessing, e.g., creating a heatmap.
            a = self.methods[aidx][2](a)
            # Store the analysis.
            analysis[0, aidx] = a[0]

        if method == "z":
            lrp = 2
        elif method == "epsilon":
            lrp = 3

        plt.figure()
        plt.imshow(analysis[0][lrp])
        plt.title("Layer-wise Relevance Propagation {}".format(method))
        print("Predicted: {}".format(text[0][0]))
        print("Actual: {}".format(text[0][3]))
        print("Logit: {}".format(text[0][1]))
        print("Probability: {}".format(text[0][2]))

        if save_path:
            plt.savefig(save_path, orientation='landscape', dpi=224)

        return text[0][0], text[0][3], text[0][1], text[0][2]

    def lrp_comparison(model, img, save_path=None):
        """

        :param model: trained model
        :param img: input image path
        :param save_path: explanation save path
        :return:
        """
        return


    def lrp_image_set(model, images, save_path=None):
        """

        :param model: trained model
        :param images: set of images to explain
        :param save_path: explanation save path
        :return:
        """
        return