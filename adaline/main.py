import numpy as np
import pandas as pd


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
    pass


if __name__ == '__main__':
    main()
