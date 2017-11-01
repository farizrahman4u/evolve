from classifier import Classifier
from activations import softmax, sigmoid
import numpy as np


class MLPClassifier(Classifier):

	def __init__(self, input_dim, num_classes, hidden_dim=50, **kwrags):
		self.input_dim = input_dim
		self.num_classes = num_classes
		self.hidden_dim = hidden_dim
		super(MLPClassifier, self).__init__(**kwrags)

	def build(self):
		self.w1 = np.random.uniform(-1, 1, (self.input_dim, self.hidden_dim))
		self.b1 = np.zeros(self.hidden_dim)
		self.w2 = np.random.uniform(-1, 1, (self.hidden_dim, self.num_classes))
		self.b2 = np.zeros(self.num_classes)
		self.weights = [self.w1, self.w2]

	def predict(self, x):
		h1 = np.dot(x, self.w1) + self.b1
		h1 = sigmoid(h1)
		h2 = np.dot(h1, self.w2) + self.b2
		y = softmax(h2)
		return y
