import random
import numpy as np
from sys import maxsize


class Perceptron:
    def __init__(self, data):
        random.shuffle(data)
        inputs = np.array([[float(x) for x in row[0:-1]] for row in data])
        self.inputs = np.hstack((inputs, [[1]] * len(inputs)))
        self.outputs = np.array([float(row[-1]) for row in data])
        self.numInputs = len(self.inputs[0])
        weights = np.array([random.uniform(0, 100) \
                                 for x in range(self.numInputs)])
        weights[-1] = -1.0
        self.weights = weights
        self.error = float(maxsize)
        self.fitHistory = []

    def predict(self, x_i):
        y = np.dot(x_i, self.weights)
        if y >= 0:
            return 1
        else:
            return 0

    def fit(self, print_weights = False, lr=1, numIters = 100):
        errorList = []
        for iter in range(numIters):
            totalError = 0.0
            for i in range(len(self.outputs)):
                pred = self.predict(self.inputs[i])
                error = self.outputs[i] - pred
                self.weights[:-1] = self.weights[:-1] + \
                               lr * error * self.inputs[i][:-1]
                self.weights[-1] = self.weights[-1] - lr * error
                totalError += abs(error)
            errorList.append(totalError)
            if totalError == 0.0:
                break
            # print("iter {} of {}".format(iter, numIters))
        self.fitHistory = errorList
        self.error = totalError
        
            
    def setError(self, e):
        self.error = e

    def printWeights(self):
        print(self.weights)

    def printError(self):
        print(self.error)

    def __str__(self):
        s = "inputs (1 sample): {}\n".format(self.inputs[0])
        s += "weights: {}\n".format(self.weights)
        s += "error: {}\n".format(self.error)
        return s
