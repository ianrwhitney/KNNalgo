from collections import defaultdict
from math import e, log

import numpy as np


class Dumb_LinearClassifier:
    def __init__(self, num_classes, image_dim):
        """Dumb_LinearClassifer has a randomly initialised weight vector (W) w/o updating via training"""
        self.num_classes = num_classes
        self.W = np.random.normal(0, .001, num_classes * image_dim).reshape(num_classes, image_dim)
        self.b = np.zeros(1)

    def train(self, X, y):
        class_losses = defaultdict(int)
        total_loss = 0

        for x_i, y_i in zip(X, y):
            ret_val = self.predict(x_i)
            # loss_val = self.svm_loss(ret_val, y_i)
            loss_val = self.cross_entropy_loss(ret_val, y_i)
            class_losses[y_i] = loss_val
            if len(class_losses) == 10:
                break

        for _, v in class_losses.items():
            total_loss += v

        return total_loss / self.num_classes

    def predict(self, X):
        ret = self.W.dot(X) + self.b
        return ret

    def svm_loss(self, preds, y):
        """ with a random initialized W vector the total loss should be approx L = num_classes - 1
            computes loss without interpretation of the values other than correct class should be higher
            than a score of all the other classes"""
        loss_val = 0
        for idx in range(len(preds)):
            if idx != y:
                loss_val += max(0, preds[idx] - preds[y] + 1)
        return loss_val

    def cross_entropy_loss(self, preds, y):
        """with a radonmly initialized W vector the total loss should be approx L = -log(num_classes)"""
        unnormalized_probabilities = e ** preds
        total_unnormalized_probabilities = sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_probabilities

        loss_val = -log(probabilities[y])
        return loss_val

