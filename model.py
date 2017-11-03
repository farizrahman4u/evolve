import numpy as np
from progressbar import ProgressBar


class Model(object):

    def __init__(self, num_pop=5, lr=0.01, dropout=0.01):
        self.num_pop = num_pop
        self.lr = lr
        self.dropout = dropout
        self.build()

    def build(self):
        raise NotImplemented

    def predict(self, x):
        # sklearn-like interface
        raise NotImplemented

    def reward(self, y_pred, y_true):
        raise NotImplemented

    def accuracy(self, y_pred, y_true):
        pass

    def loss(self, y_pred, y_true):
        pass

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        acc = self.accuracy(y_pred, y)
        loss = self.loss(y_pred, y)
        return {'accuracy': acc, 'loss': loss}

    def __call__(self, *args, **kwargs):
        # functional interface
        return self.predict(*args, **kwargs)


    def train_on_batch(self, x, y):
        mutations = [[np.random.uniform(-self.lr, self.lr, w.shape) for w in self.weights] for _ in range(self.num_pop)]
        dropout_mask = [np.random.binomial(1, self.dropout, w.shape) for w in self.weights]
        for M in mutations:
            for m, d in zip(M, dropout_mask):
                m *= d
        rewards = np.zeros(self.num_pop)
        current_reward = self.reward(self.predict(x), y).mean()
        for j, m in enumerate(mutations):
            for w1, w2 in zip(m, self.weights):
                w2 += w1
            y_hat = self.predict(x)
            for w1, w2 in zip(m, self.weights):
                w2 -= w1
            rewards[j] = self.reward(y_hat, y).mean()
        rewards -= current_reward
        std = np.std(rewards)
        if std == 0:
            rewards = np.zeros_like(rewards)
        else:
            rewards /= std
        for r, m in zip(rewards, mutations):
            for w1, w2 in zip(m, self.weights):
                w2 += r * w1
        #w += update

    def _train_on_batch(self, x, y):
        w_idx = np.random.randint(0, len(self.weights))
        W = self.weights[w_idx]
        for w in [W]:
            mutations = np.random.uniform(-self.lr, self.lr,(self.num_pop, ) + w.shape)
            dropout_mask = np.random.binomial(1, self.dropout, w.shape)
            mutations *= dropout_mask
            rewards = np.zeros(self.num_pop)
            current_reward = self.reward(self.predict(x), y).mean()
            for j, m in enumerate(mutations):
                w += m
                y_hat = self.predict(x)
                w -= m
                rewards[j] = self.reward(y_hat, y).mean()
            rewards -= current_reward
            std = np.std(rewards)
            if std == 0:
                rewards = np.zeros_like(rewards)
            else:
                rewards /= std
            update = np.tensordot(rewards, mutations, (0, 0))
            w += update

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
                print(self.evaluate(*validation_data))

    def get_weights(self):
        return [w.copy() for w in self.weights]

    def set_weights(self, weights):
        self.weights = [w.copy() for w in weights]

