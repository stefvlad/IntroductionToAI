import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx]
                    , label=cl)


class Adaline:
    def __init__(self, learning_rate=0.01, learning_iterations=100, random_state=3):
        self.learning_rate = learning_rate
        self.learning_iterations = learning_iterations
        self.random_state = random_state
        self.weights = np.array([])  # size = n_columns + 1
        self.cost_func_values = []  # size = learning_iterations

    def activation_func(self, input_data):
        return input_data

    def net_input(self, inputs):  # vector - size = inputs.shape[0] = n_rows
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

    def predict(self, input_data):  # threshold function
        return np.where(self.activation_func(self.net_input(input_data)) >= 0, 1, -1)

    def train_adaline(self, inputs, targets):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=inputs.shape[1] + 1)

        for _ in range(self.learning_iterations):
            net_input = self.net_input(inputs)
            output = self.activation_func(net_input)
            errors = (targets - output)
            self.weights[1:] += -self.learning_rate * inputs.T.dot(-errors)
            self.weights[0] += -self.learning_rate * (-errors.sum())
            cost = (errors ** 2).sum() / 2.0
            self.cost_func_values.append(cost)


def main():
    url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    downloaded_df = pd.read_csv(url, names=column_names)

    x = downloaded_df.iloc[0: 100, [0, 2]].values
    y = downloaded_df.iloc[0: 100, 4].values

    x_std = np.copy(x)
    x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()  # input normalization, standardization
    x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()  # input normalization, standardization

    y = np.where(y == 'Iris-setosa', 1, -1)

    ada_1 = Adaline(learning_rate=0.01, learning_iterations=15)
    ada_1.train_adaline(x_std, y)

    plot_decision_regions(x_std, y, classifier=ada_1)
    plt.title('Adaline - gradient descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()

    plt.plot(range(1, len(ada_1.cost_func_values) + 1), ada_1.cost_func_values, marker='o')
    plt.xlabel('epochs')
    plt.ylabel('sum_squared_errors')
    plt.tight_layout()

    plt.show()

    # ada_1 = Adaline(learning_rate=0.01, learning_iterations=20)
    # ada_1.train_adaline(x_std, y)

    # ada_2 = Adaline(learning_rate=0.0001, learning_iterations=20)
    # ada_2.train_adaline(x_std, y)

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # ax[0].plot(range(1, len(ada_1.cost_func_values) + 1), ada_1.cost_func_values, marker='o')
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('sum_squared_errors')
    # ax[0].set_title('ADALINE - learning_rate = 0.01')
    #
    # ax[1].plot(range(1, len(ada_2.cost_func_values) + 1), ada_2.cost_func_values, marker='o')
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('sum_squared_errors')
    # ax[1].set_title('ADALINE - learning_rate = 0.0001')


if __name__ == '__main__':
    main()
