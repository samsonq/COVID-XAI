import numpy as np
from keras import backend as K
from keras.datasets import mnist

import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis


############################
# Post Processing Utility
############################


def postprocess(X):
    X = X.copy()
    X = iutils.postprocess_images(X)
    return X


def bk_proj(X):
    return ivis.graymap(X)


def heatmap(X):
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X))
