# Base class for classifiers

from model import Model
import numpy as np


class Classifier(Model):

    def reward(self, y_pred, y_true):
    	return -self.loss(y_pred, y_true)

    def accuracy(self, y_pred, y_true):
        labels_pred = np.argmax(y_pred, -1)
        labels_true = np.argmax(y_true, -1)
        num_correct_labels = np.sum(labels_pred == labels_true)
        num_correct_labels = float(num_correct_labels)
        return num_correct_labels / len(y_pred)

    def loss(self, y_pred, y_true):
        # cross entropy
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
