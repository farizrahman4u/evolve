import numpy as np

def to_categorical(labels, num_classes):
	n = len(labels)
	y = np.zeros((n, num_classes))
	y[np.arange(n), labels] = 1
	return y
