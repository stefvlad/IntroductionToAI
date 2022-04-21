import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None


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


def main():
    url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    downloaded_df = pd.read_csv(url, names=column_names)

    my_df = downloaded_df[['sepal_length', 'sepal_width', 'species']]

    setosa_virginica = my_df.loc[(my_df['species'] == 'Iris-setosa') | (my_df['species'] == 'Iris-virginica')]

    setosa_virginica['species'] = setosa_virginica['species'].map({'Iris-setosa': 1, 'Iris-virginica': -1})

    inputs = setosa_virginica.loc[:, ['sepal_length', 'sepal_width']].values
    targets = setosa_virginica.loc[:, ['species']].values

    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, train_size=0.25, random_state=4)

    p = Perceptron()

    p.train_perceptron(x_train, y_train)

    # print(p.errors)

    count = 0
    for i in range(len(x_test)):
        if p.predict(x_test[i]) == y_test[i]:
            count += 1
        # print('perceptron result = %d\t real result = %d' % (p.predict(x_test[i]), y_test[i]))

    print("total: %d\t correct results: %d" % (len(x_test), count))
    print("accuracy score: %f" % (count / len(x_test)))

    plt.xlabel('sepal_length(cm)')
    plt.ylabel('sepal_width(cm)')

    for i in range(len(x_test)):
        if y_test[i] == 1:
            color = 'blue'
        else:
            color = 'green'

        plt.scatter(x_test[i:, 0], x_test[i:, 1], c=color, marker='o')

    plt.legend(['setosa', 'virginica'], loc="upper right")
    leg = plt.gca().get_legend()
    leg.legendHandles[0].set_color('blue')
    leg.legendHandles[1].set_color('green')
    plt.grid(True)

    p1 = [min(x_test[:, 0]), max(x_test[:, 0])]
    m = -p.weights[1] / p.weights[2]
    c = -p.weights[0] / p.weights[2]
    p2 = [p1[0] * m + c, p1[1] * m + c]
    plt.plot(p1, p2)

    plt.show()


if __name__ == '__main__':
    main()
