import numpy as np

def sigmoid(x):
	return 1. / (1. + np.exp(-x))


def softmax(x):
	y = np.exp(x - np.max(x, axis=-1, keepdims=True))
	s = np.sum(y, axis=-1, keepdims=True)
	y /= s
	return y
