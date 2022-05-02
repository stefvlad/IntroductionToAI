import numpy as np
import pandas as pd


# import matplotlib.pyplot as plt


def predict_iris(perc0, perc1, perc2, data):
    if perc0.predict(data) == 1:
        print('setosa')
    elif perc1.predict(data) == 1:
        print('versicolor')
    elif perc2.predict(data) == 1:
        print('virginica')


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

    y_perc0 = downloaded_df.iloc[0:150, 4].values
    y_perc1 = downloaded_df.iloc[0:150, 4].values
    y_perc2 = downloaded_df.iloc[0:150, 4].values

    y_perc0 = np.where(y_perc0 == 'Iris-setosa', 1, -1)
    y_perc1 = np.where(y_perc1 == 'Iris-versicolor', 1, -1)
    y_perc2 = np.where(y_perc2 == 'Iris-virginica', 1, -1)

    x = downloaded_df.iloc[0:150, [0, 1, 2, 3]].values

    # plt.xlabel('sepal length(cm)')
    # plt.ylabel('petal length(cm)')

    # 'ingerencja w dane'
    for index in range(100, 150):
        x[index][2] -= 4.9
        x[index][3] -= 1.4

    # plt.scatter(x[:50, 0], x[:50, 1], c='red', marker='o', label='setosa')
    # plt.scatter(x[50:100, 0], x[50:100, 1], c='blue', marker='x', label='versicolor')
    # plt.scatter(x[100:, 0], x[100:, 1], c='green', marker='>', label='virginica')

    perc0 = Perceptron(learning_rate=0.001, learning_iterations=150)
    perc0.train_perceptron(x, y_perc0)

    perc1 = Perceptron(learning_rate=0.001, learning_iterations=150)
    perc1.train_perceptron(x, y_perc1)

    perc2 = Perceptron(learning_rate=0.001, learning_iterations=150)
    perc2.train_perceptron(x, y_perc2)

    # errors_per_epoch_0 = np.array(perc0.errors)
    # errors_per_epoch_1 = np.array(perc1.errors)
    # errors_per_epoch_2 = np.array(perc2.errors)

    # total_errors = errors_per_epoch_0 + errors_per_epoch_1 + errors_per_epoch_2

    # plt.plot(range(1, len(total_errors) + 1), total_errors, marker='x')

    # plt.legend()
    # plt.grid(True)

    # plt.show()

    print('sepal length   sepal width   petal length   petal width')
    while True:
        sepal_length, sepal_width, petal_length, petal_width = map(float, input().split())

        data = np.array([sepal_length, sepal_width, petal_length, petal_width])

        predict_iris(perc0, perc1, perc2, data)


if __name__ == '__main__':
    main()
