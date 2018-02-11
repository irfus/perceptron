import sys
from perceptron import Perceptron

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = f.read().split('\n')
        data = [row.split('\t') for row in data]
    weights = sys.argv[2:]
    weights = [float(weight) for weight in weights]
    model = Perceptron(data, weights)
    model.test()
