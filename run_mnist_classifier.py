from classifier import Classifier
from mnist import load_data
from np_utils import to_categorical
import numpy as np


batch_size = 32
epochs = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

clf = Classifier(784, 10)   # 784 is the number of pixels in an image

clf.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

accuracy = clf.evaluate(x_test, y_test) * 100

print('Accuracy = ' + str(accuracy) + '%')
