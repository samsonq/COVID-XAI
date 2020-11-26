import os
import numpy as np
import cv2
from tqdm import tqdm
from config import LABELS, IMG_SIZE, GRAYSCALE, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH
from utils import create_preprocessing_f


class DataLoader:
    # TODO: implement image to id tracker for LRP/Lime plot naming
    """

    """
    def __init__(self, labels=LABELS, img_size=IMG_SIZE, gray=GRAYSCALE):
        self.labels = labels
        self.img_size = img_size
        self.gray = gray
        self.num_classes = len(labels)
        self.label_to_class_name = {j: i for i, j in enumerate(labels)}

    def load_data(self, data_dir):
        """

        :param data_dir:
        :return:
        """
        data = []
        for label in self.labels:
            # Class Balancing
            if label == "BACTERIA" and (data_dir == "./data/train" or data_dir == "./data/val"):
                label = "BACTERIA_SUB"

            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label) if label != "BACTERIA_SUB" else 0
            for img in os.listdir(path):
                try:
                    if self.gray:
                        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    else:
                        img_arr = cv2.imread(os.path.join(path, img))
                    resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))  # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)

    def preprocessing(self, train, val, test):
        """

        :param train:
        :param val:
        :param test:
        :return:
        """
        x_train = []
        y_train = []

        x_val = []
        y_val = []

        x_test = []
        y_test = []

        for feature, label in tqdm(train):
            x_train.append(feature)
            y_train.append(label)

        for feature, label in tqdm(test):
            x_test.append(feature)
            y_test.append(label)

        for feature, label in tqdm(val):
            x_val.append(feature)
            y_val.append(label)

        x_train = np.array(x_train, dtype="float")
        x_val = np.array(x_val, dtype="float")
        x_test = np.array(x_test, dtype="float")

        dim = 1 if self.gray else 3
        x_train = x_train.reshape(-1, self.img_size, self.img_size, dim)
        y_train = np.array(y_train)

        x_val = x_val.reshape(-1, self.img_size, self.img_size, dim)
        y_val = np.array(y_val)

        x_test = x_test.reshape(-1, self.img_size, self.img_size, dim)
        y_test = np.array(y_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def prepare_data(self, train_path=TRAIN_DATA_PATH, val_path=VAL_DATA_PATH, test_path=TEST_DATA_PATH):
        train = self.load_data(train_path)
        test = self.load_data(test_path)
        val = self.load_data(val_path)

        x_train, y_train, x_val, y_val, x_test, y_test = self.preprocessing(train, val, test)
        preprocess, revert_preprocessing = create_preprocessing_f(x_train)

        data = (
            preprocess(x_train), y_train,
            preprocess(x_val), y_val
        )

        preprocess_test, revert_preprocessing_test = create_preprocessing_f(x_test)
        x_test = preprocess_test(x_test)

        return data, revert_preprocessing, x_test, revert_preprocessing_test, y_test
