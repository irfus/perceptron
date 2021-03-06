import random
import numpy as np
import sys


class Perceptron:
    def __init__(self, data, weights = None):
        random.shuffle(data)
        inputs = np.array([[float(x) for x in row[0:-1]] for row in data])
        self.inputs = np.hstack((inputs, [[1]] * len(inputs))) # Append 1 to each input row, for the bias weight
        self.outputs = np.array([float(row[-1]) for row in data])
        self.numInputs = len(self.inputs[0])
        if weights == None:
            weights = np.array([random.uniform(0, 100) \
                                     for x in range(self.numInputs)])
            weights[-1] = -1 # set initial value of bias weight
        self.weights = weights
        self.error = float(sys.maxsize) # initialise error to some very high value
        self.smallestError = self.error
        self.bestWeights = self.weights
        self.fitHistory = []

    def predict(self, x_i):
        y = np.dot(x_i, self.weights) # Activation function is the dot product of input vector and weight vector
        return 1 if y > 0 else 0

    def fit(self, lr=1, numIters = 100, breakSoon=True):
        errorList = []
        for iter in range(numIters):
            totalError = 0.0
            for i in range(len(self.outputs)):
                pred = self.predict(self.inputs[i])
                error = self.outputs[i] - pred # Error is the difference between true and predicted class
                self.weights = self.weights + \
                               lr * error * self.inputs[i] # multiplying with the error yields a positive or negative adjustment depending on a positive or negative prediction error
                totalError += abs(error)
            
            self.saveBestFit(self.weights, totalError)
            if breakSoon:
                if totalError == 0.0:
                    break
            self.printWeights()
            errorList.append(totalError)

        self.fitHistory = errorList # Store error history for convenient plotting
        self.error = totalError
        
    def saveBestFit(self, w, e): # Store the best performing weights for reuse
        if e < self.smallestError:
            self.smallestError = e
            self.bestWeights = w

    def printWeights(self):
        print("\t".join(map(str, self.weights)), file=sys.stderr)

    def test(self): # Ideally we should split data into train/test sets to feed this method. For now, just use the data passed during initialisation.
        e = 0.0
        for i in range(len(self.inputs)):
            pred = self.predict(self.inputs[i])
            e += self.outputs[i] - pred
        print(e, file=sys.stdout)
        
    def __str__(self):
        s = "inputs (1 sample): {}\n".format(self.inputs[0])
        s += "weights: {}\n".format(self.weights)
        s += "error: {}\n".format(self.error)
        return s
