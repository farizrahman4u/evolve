from classifier import Classifier
from activations import softmax
import numpy as np


class SimpleClassifier(Classifier):

    def __init__(self, input_dim, num_classes, **kwargs):
        self.input_dim = input_dim
        self.num_classes = num_classes
        super(SimpleClassifier, self).__init__(**kwargs)

    def build(self):
        w = np.zeros((self.input_dim, self.num_classes))
        self.weights = [w]

    def predict(self, x):
    	w = self.weights[0]
        h = np.dot(x, w)
       	return softmax(h)
