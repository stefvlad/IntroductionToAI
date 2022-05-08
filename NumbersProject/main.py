import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_letters(input_data):
    fig, ax = plt.subplots(nrows=2, ncols=13, figsize=(13, 3))

    for i in range(2):
        for j in range(13):

            letter = input_data[i * 13 + j]

            for y in range(7):
                for x in range(5):
                    if letter[y * 5 + x] == 1:
                        ax[i, j].scatter(x, y, marker='s', s=90)

            ax[i, j].invert_yaxis()
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])

    plt.show()


def main():
    df = pd.read_csv('data.csv', header=None)

    X = df.iloc[0:26, 0:35].values
    Y = df.iloc[0:26, 35:].values

    show_letters(X)


if __name__ == '__main__':
    main()
