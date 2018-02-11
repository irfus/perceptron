import sys
from perceptron import Perceptron

if __name__ == "__main__":
    iters = int(sys.argv[1])
    lr = float(sys.argv[2])
    with open(sys.argv[3]) as f:
        data = f.read().split('\n')
        data = [row.split('\t') for row in data]
    model = Perceptron(data)

    model.fit(lr, iters, True)
    print("\t".join(map(str, model.bestWeights)), file=sys.stdout)
