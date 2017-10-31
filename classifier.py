import numpy as np
from progressbar import ProgressBar

class Classifier(object):

    def __init__(self, input_dim, num_classes, use_bias=False):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.build()

    def build(self):
        self.W = np.zeros((self.input_dim + int(self.use_bias), self.num_classes))
        #self.W = np.random.uniform(-.1, .1, (self.input_dim, self.num_classes))
        self.previous_update = np.zeros_like(self.W)

    def predict(self, x, w=None):
        if w is None:
            w = self.W
        if self.use_bias:
            b = w[-1]
            w = w[:-1]
            h = np.dot(x, w) + b
        else:
            h = np.dot(x, w)
        y = np.exp(h - np.max(h, axis=1, keepdims=True))
        s = np.sum(y, axis=1, keepdims=True)
        y /= s
        return y

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def reward(self, y_pred, y_true):
        labels = np.argmax(y_true, axis=-1)
        return np.diag(y_pred.T[labels])

    def train_on_batch(self, x, y):
        num_pop = 5
        lr = 0.01
        mutations = np.random.uniform(-lr, lr,(num_pop, ) + self.W.shape) * np.random.binomial(1, 0.01, self.W.shape)  # num_pop, input_dim, num_classes
        #mutations.append(self.previous_update)
        rewards = np.zeros(num_pop)
        current_reward = self.reward(self.predict(x), y).mean()
        for i, m in enumerate(mutations):
            w = m + self.W
            y_hat = self.predict(x, w)
            rewards[i] = self.reward(y_hat, y).mean()
        rewards -= current_reward#np.mean(rewards)
        
        std = np.std(rewards)
        if std == 0:
            rewards = np.zeros_like(rewards)
        else:
            rewards /= std
        #rewards *= rewards < 0
        '''
        update = np.zeros_like(self.W)
        max_idx = np.argmax(rewards)
        max_reward = rewards[max_idx]
        if max_reward > 0:
            update = mutations[max_idx] * max_reward
        min_idx = np.argmin(rewards)
        min_reward = rewards[min_idx]
        if min_reward < 0:
            pass#update -= mutations[min_idx] * np.abs(min_reward)
        '''
        update = np.tensordot(rewards, mutations, (0, 0))
        #update = update + 0.1 * self.previous_update
        #self.previous_update = update
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
