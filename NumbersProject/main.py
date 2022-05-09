import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def damage(X, percent, seed=1):
    rgen = np.random.RandomState(seed)
    result = np.array(X)
    count = int(X.shape[1] * percent / 100)

    for index_example in range(len(X)):
        order = np.sort(rgen.choice(X.shape[1], count, replace=False))
        for index_pixel in order:
            result[index_example][index_pixel] *= -1

    return result


class Perceptron:
    def __init__(self, learning_rate=0.01, learning_iterations=100, random_state=3):
        self.learning_rate = learning_rate
        self.learning_iterations = learning_iterations
        self.random_state = random_state
        self.errors = []
        self.weights = np.array([])

    def predict(self, input_data):
        return np.where((np.dot(input_data, self.weights[1:]) + self.weights[0]) >= 0.0, 1, -1)

    def train_perceptron(self, inputs, targets):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=inputs.shape[1] + 1)

        for _ in range(self.learning_iterations):
            n_error = 0
            for input_i, target_i in zip(inputs, targets):
                delta_w = self.learning_rate * (target_i - self.predict(input_i))
                self.weights[0] += delta_w

                for i in range(1, len(self.weights)):
                    self.weights[i] += delta_w * input_i[i - 1]

                if delta_w != 0.0:
                    n_error += 1
            self.errors.append(n_error)
        return self


class Network:
    def __init__(self, eta=0.05, n_iter=10, random_state=1):
        self.perceptrons = []
        self.predicted = []
        for i in range(10):
            self.perceptrons.append(Perceptron(eta, n_iter, random_state))

    def fit(self, X, Y):
        for i in range(10):
            self.perceptrons[i].train_perceptron(X, Y[i])

    def errors(self):
        total_errors = np.zeros(len(self.perceptrons[0].errors), dtype=int)

        for i in range(10):
            total_errors += np.array(self.perceptrons[i].errors)

        print(total_errors)

        plt.plot(range(1, len(total_errors) + 1), total_errors, marker='x')

        plt.show()

    def show(self, X):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5.25, 3))

        for i in range(2):
            for j in range(5):

                letter = X[i * 5 + j]

                for y in range(7):
                    for x in range(5):
                        if letter[y * 5 + x] == 1:
                            ax[i, j].scatter(x, y, marker='s', s=90)

                ax[i, j].invert_yaxis()
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])

        plt.show()

    def predict(self, X):
        if len(self.predicted) == 0:
            for i in range(len(self.perceptrons)):
                self.predicted.append(self.perceptrons[i].predict(X))
        else:
            for i in range(len(self.perceptrons)):
                self.predicted[i] = self.perceptrons[i].predict(X)

        print(self.predicted)

    def misclassified(self, Y):
        print("misclassified examples: %d" % (np.array(self.predicted) != Y).sum())


def main():
    df = pd.read_csv('data.csv', header=None)

    X = df.iloc[[5, 10, 11, 12, 14, 16, 17, 19, 22, 24], 0:35].values
    Y = df.iloc[0:10, 35:45].values

    network = Network()
    network.show(X)

    network.fit(X, Y)
    network.errors()

    network.predict(X)
    network.misclassified(Y)

    network.predict(damage(X, 5))
    network.misclassified(Y)

    network.predict(damage(X, 15))
    network.misclassified(Y)

    network.predict(damage(X, 40))
    network.misclassified(Y)


if __name__ == '__main__':
    main()
