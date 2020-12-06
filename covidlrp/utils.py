import os
import random
import shutil
import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from config import LABELS, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


config = load_config("./config.yaml")


def print_class_distribution(train_path=TRAIN_DATA_PATH, val_path=VAL_DATA_PATH, test_path=TEST_DATA_PATH):
    """

    :param train_path:
    :param val_path:
    :param test_path:
    """
    for dataset, name in {train_path: "Training", val_path: "Validation", test_path: "Test"}.items():
        for label in LABELS:
            print("{} {} instances:".format(name, label), len(os.listdir(os.path.join(dataset, label))))
    return


def train_val_split(train_path=TRAIN_DATA_PATH, val_path=VAL_DATA_PATH, split_size=0.2):
    """

    :param train_path:
    :param val_path:
    :param split_size:
    """
    for label in LABELS:
        train_size = len(os.listdir(os.path.join(train_path, label)))
        val_size = len(os.listdir(os.path.join(val_path, label)))
        move_size = split_size * (train_size + val_size)
        if val_size > move_size:
            print("Current validation set too large for {}.".format(label))
            continue
        for i in random.sample(os.listdir(os.path.join(train_path, label)), move_size - val_size):
            shutil.move(os.path.join(train_path, label, i), os.path.join(val_path, label, i))
    print_class_distribution(train_path=train_path, val_path=val_path)
    return


def visualize_distribution(data):
    """

    :param data:
    :return:
    """
    l = []
    for i in data:
        if i[1] == 0:
            l.append("Bacteria")
        elif i[1] == 1:
            l.append("Virus")
        else:
            l.append("Normal")

    sns.set_style("darkgrid")
    sns.countplot(l)
    plt.show()


def visualize_samples(data, labels=LABELS, samples=1):
    """

    :param data:
    :param labels:
    :param samples:
    :return:
    """
    for _ in range(samples):
        x = random.randint(0, data.size-1)
        plt.figure(figsize=(5, 5))
        plt.imshow(data[x][0], cmap='gray')
        plt.title(labels[data[x][1]])


def create_preprocessing_f(X, input_range=[0, 1]):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    :param X:
    :param input_range:
    :return:
    """
    if len(input_range) != 2:
        raise ValueError(
            "Input range must be of length 2, but was {}".format(
                len(input_range)))
    if input_range[0] >= input_range[1]:
        raise ValueError(
            "Values in input_range must be ascending. It is {}".format(
                input_range))

    a, b = X.min(), X.max()
    c, d = input_range

    def preprocessing(X):
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= (b-a)
        X *= (d-c)
        # shift to desired output range
        X += c
        return X

    def revert_preprocessing(X):
        X = X - c
        X /= (d-c)
        X *= (b-a)
        X += a
        return X

    return preprocessing, revert_preprocessing
