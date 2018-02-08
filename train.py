import numpy as np
from perceptron import Perceptron
from matplotlib import pyplot as plt

%matplotlib inline

model = Perceptron(data)
print(model, "\n")
model.fit(numIters = 100)
plt.plot(model.fitHistory)
plt.xlabel("Iterations")
plt.ylabel("Error")

print(model)
