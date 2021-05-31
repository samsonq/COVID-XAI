import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries


class LIME:
    """
    Local Interpretable Model-agnostic Explanations
    """
    def __init__(self, model, num_samples=1000):
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.num_samples = num_samples
        self.explainer = lime_image.LimeImageExplainer()

    def lime(self, img, label=None, save_path=None):
        explanation = self.explainer.explain_instance(img.astype('double'),
                                                      self.model.predict,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=self.num_samples)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=True,
                                                    num_features=5,
                                                    hide_rest=False)
        plt.figure()
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        if label:
            plt.title("Pred: {}, Actual: {}".format(self.model.predict_on_batch(img)[0], label))
        else:
            plt.title("LIME Output")
        if save_path:
            plt.savefig(save_path)
        return mask

    def lime_comparison(self):
        return

    def lime_image_set(self):
        return


def show_lime(model, img, data_loader, save_name=None):
    label_dict = data_loader.label_to_class_name
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img.astype('double'),
                                             model.predict,
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=False)
    # TODO:
    label = data_loader.img_to_label[img]

    fig, ax = plt.subplots(1, 1)
    plt.title("Pred: {}, Actual: {}".format(label_dict[np.argmax(model.predict_on_batch(img)[0])], label_dict[y]))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    if save_name is not None:
        plt.savefig(save_name)
