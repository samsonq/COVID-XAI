import os, json
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries


def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transf


def get_input_tensors(img):
    transf = get_input_transform()
    # unsqueeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


class TorchLIME:
    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, samples=1000):
        self.explanation = self.explainer.explain_instance(num_samples=samples)

    def show_explanation(self, positive_only=True, features=5, hide_img=False):
        """

        :param positive_only:
        :param features:
        :param hide_img:
        :return:
        """
        temp, mask = self.explanation.get_image_and_mask(self.explanation.top_labels[0],
                                                         positive_only=positive_only, num_features=features,
                                                         hide_rest=hide_img)
        img_boundry = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry)

