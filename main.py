from sklearn.model_selection import train_test_split
from NearestNeighbor import KNN
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from LinearClassifiers import Dumb_LinearClassifier


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def dist(x1, x2):
    return np.abs(np.sum(x1 - x2))


if __name__ == '__main__':
    mnist = load_digits()
    X = mnist.data
    y = mnist.target
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=123)
    # accuracies = []
    # formulas = [euc_dist, dist]
    # k_values = range(1, 1000, 10)
    # for k in k_values:
    #     model = KNN(euc_dist, k)
    #     model.train(train_X, train_y)
    #     pred = model.predict(test_X)
    #     acc = accuracy_score(test_y, pred) * 100
    #     accuracies.append(acc)
    #     print(f"K: {k}  accuracy: {acc:.2f}")
    #
    # plt.plot(k_values, accuracies)
    # plt.xlabel("K Values")
    # plt.ylabel("Accuracy")
    # plt.show()
    linear_classifier = Dumb_LinearClassifier(10, 8 * 8)
    print(train_y[0])
    print(linear_classifier.train(train_X, train_y))
    pred = linear_classifier.predict(test_X[0])
    # print(pred)
