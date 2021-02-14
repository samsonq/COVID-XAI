import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.utils import to_categorical
from config import LABELS


def plot_acc_loss(history):
    """
    Plot model training/validation loss and accuracy across epochs trained.
    :param history: model training history
    :return: plot figure object
    """
    epochs = [i for i in range(len(history))]  # number of train epochs

    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['acc']
    train_loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax[1].set_title('Training & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.show()
    return fig


def classification_metrics(model, X, y):
    """
    Display precision, recall, f1-score classification metrics for model on data.
    :param model: model to make predictions on images
    :param X: features
    :param y: labels
    :return: classification report
    """
    predictions = model.predict(X).argmax(axis=-1)
    predictions = predictions.reshape(1, -1)[0]
    return classification_report(y, predictions, target_names=LABELS)


def plot_confusion(model, X, y):
    """
    Plot confusion matrix of model classifications on data.
    :param model: model to make predictions on images
    :param X: features
    :param y: labels
    """
    predictions = model.predict(X).argmax(axis=-1)
    predictions = predictions.reshape(1, -1)[0]

    correct = np.nonzero(predictions == y)[0]
    incorrect = np.nonzero(predictions != y)[0]
    print("Correct:", len(correct))
    print("Incorrect:", len(incorrect))

    y_pred = model.predict(X)
    # to get the prediction, we pick the class with with the highest probability
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(to_categorical(y, len(LABELS)), axis=1)

    conf_mtx = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(conf_mtx, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(3), LABELS, fontsize=16)
    plt.yticks(range(3), LABELS, fontsize=16)
    plt.show()
