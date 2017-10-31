import numpy as np
from progressbar import ProgressBar

class Classifier(object):

    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.W = np.zeros((self.input_dim, self.num_classes))
        #self.W = np.random.uniform(-.1, .1, (self.input_dim, self.num_classes))
        self.previous_update = np.zeros_like(self.W)

    def predict(self, x):
        h = np.dot(x, self.W)
        y = np.exp(h - np.max(h, axis=1, keepdims=True))
        s = np.sum(y, axis=1, keepdims=True)
        y /= s
        return y

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def reward(self, y_pred, y_true):
        labels = np.argmax(y_true, axis=-1)
        return np.diag(y_pred.T[labels])


    def _train_on_batch(self, x, y):
        num_pop = 10
        lr = 0.01
        mutations = np.random.normal(-0.1, 0.1, (num_pop,) + self.W.shape)  # num_pop, input_dim, num_classes
        W = mutations + self.W  # num_pop, input_dim, num_classes
        h = np.tensordot(x, W, (1, 1))  # batch_size, num_pop, num_classes
        h = np.transpose(h, (1, 0, 2))  # num_pop, batch_size, num_classes
        h = np.reshape(h, (-1, self.num_classes))  # num_pop * batch_size, num_classes
        y_hat = np.exp(h - np.max(h, axis=1, keepdims=True))  # num_pop * batch_size, num_classes
        s = np.sum(y_hat, axis=1, keepdims=True)  # num_pop * batch_size, 1
        y_hat /= s  # num_pop * batch_size, num_classes
        y = np.tile(y, (num_pop, 1)) # num_pop * batch_size, num_classes
        rewards = self.reward(y_hat, y)  # num_pop * batch_size
        rewards = np.reshape(rewards, (num_pop, -1))
        rewards = np.mean(rewards, axis=1)  # num_pop
        rewards -= np.mean(rewards)
        std = np.std(rewards)
        if std == 0:
            rewards = np.zeros_like(rewards)
        else:
            rewards /= std
        update = np.tensordot(rewards, mutations, (0, 0))
        self.W += lr * update

    def train_on_batch(self, x, y):
        num_pop = 10
        lr = 0.01
        mutations = [np.random.normal(-lr, lr, self.W.shape) for _ in range(num_pop)]  # num_pop, input_dim, num_classes
        #mutations.append(self.previous_update)
        rewards = np.zeros(num_pop)
        current_reward = self.reward(self.predict(x), y).mean()
        for i, m in enumerate(mutations):
            W = m + self.W
            h = np.dot(x, W)
            y_hat = np.exp(h - np.max(h, axis=1, keepdims=True))
            s = np.sum(y_hat, axis=1, keepdims=True)
            y_hat /= s
            rewards[i] = self.reward(y_hat, y).mean()
        rewards -= current_reward#np.mean(rewards)
        #rewards *= rewards > 0
        
        std = np.std(rewards)
        if std == 0:
            rewards = np.zeros_like(rewards)
        else:
            rewards /= std
        
        update = np.tensordot(rewards, mutations, (0, 0))
        #update = update + 0.1 * self.previous_update
        self.previous_update = update
        self.W += update

    def fit(self, x, y, batch_size=32, epochs=10, validation_data=None):
        num_samples = len(x)
        for epoch in range(epochs):
            print('')
            print ('Epoch ' + str(epoch + 1) + ': ')
            idx = 0
            pbar = ProgressBar(num_samples)
            while idx < num_samples:
                x_batch = x[idx : idx + batch_size]
                y_batch = y[idx : idx + batch_size]
                self.train_on_batch(x_batch, y_batch)
                pbar.add(batch_size)
                idx += batch_size
            if validation_data:
                print('')
                print('Accuracy :' + str(self.evaluate(*validation_data) * 100) + '%')

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        predcited_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        num_correct_labels = np.sum(true_labels == predcited_labels)
        accuracy = float(num_correct_labels) / len(x)
        return accuracy
