# Evolve : Evolutionary Strategies for Training Neural Networks [WIP]

## You have just found Evolve

### Good bye backprop!

A humble attempt to train neural networks without backpropogation.

```shell
git clone https://www.github.com/farizrahman4u/evolve.git
cd evolve
python run_mnist_classifier.py
```


#### Example

##### Training a 2 layer Neural Network on MNIST:

Write your NN class:

```python
# Import classifier base class.
# It contains methods for cross entropy, classification accuracy etc.
from classifier import Classifier

# import activations
from activations import sigmoid, softmax


class MLPClassifier(Classifier):
    def __init__(self, input_dim, num_classes, hidden_dim, **kwargs):
        self.input_dim = input_dim  # size of input vectors
        self.num_classes = num_classes  # number of categories
        self.hidden_dim = hidden_dim  # size of the hidden layer
        super(MLPClassifer, self).__init__(**kwargs)  # call base class constructor

    def build(self):
        # Method for initializing weights
        self.w1 = np.random.uniform(-1, 1, (self.input_dim, self.hidden_dim))  # layer 1 matrix
        self.b1 = np.zeros(self.hidden_dim)  # layer 1 bias
        self.w2 = np.random.uniform(-1, 1, (self.hidden_dim,  self.num_classes))  # layer 2 matrix
        self.b2 = np.zeros(self.num_classes)  # layer 2 bias
        # weights wont be trained unless they are added to self.weights list
        self.weights = [self.w1, self.b1, self.w2, self.b2]

    def predict(self, x):
        # Your forward pass logic lives here
        # vanilla 2 layer NN
        h1 = np.dot(x, self.w1)
        h1 = sigmoid(h1)
        h2 = np.dot(h1, self.w2)
        y = softmax(h2)
        return y

```

Load and process MNIST data:

```python
import mnist
import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
```

Train!!

```python
clf = MLPClassifier(784, 10)
clf.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))
```

You can do inference using the functional interface:

```
x_random = np.random.random((32, 784))  # 32 random inputs
y = clf(x)  # can also do y = clf.predict(x)
print(y.shape)  # >>>(32, 10)
```
