import os
import numpy as np
import cv2
from tqdm import tqdm
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
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
        self.data = None
        self.preprocess = None

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

        preprocess_test, revert_preprocessing_test = create_preprocessing_f(x_test)

        # One-hot Encoding
        y_train = to_categorical(y_train, len(LABELS))
        y_val = to_categorical(y_val, len(LABELS))
        y_test = to_categorical(y_test, len(LABELS))

        data = (
            preprocess(x_train), y_train,
            preprocess(x_val), y_val,
            preprocess_test(x_test), y_test
        )
        self.data = data
        self.preprocess = [revert_preprocessing, revert_preprocessing_test]

        return data, [revert_preprocessing, revert_preprocessing_test]

    def data_aug(self):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.2,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(self.data[0])
        return datagen
