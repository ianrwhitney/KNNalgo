from collections import defaultdict
import numpy as np
from operator import itemgetter


class KNN:
    def __init__(self, formulas, K=3):
        self.K = K
        self.formulas = formulas

    def train(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = np.array([self.formulas(X_test[i], x_t) for x_t in self.X_train])
            distances_sorted = np.asarray(distances).argsort()[:self.K]
            neighbors = defaultdict(int)
            for idx in distances_sorted:
                neighbors[self.Y_train[idx]] += 1
            predictions.append(max(neighbors.items(), key=itemgetter(1))[0])
        return predictions
