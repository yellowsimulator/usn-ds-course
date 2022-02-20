import numpy as np
import matplotlib.pyplot as plt



# relu activation function
def relu(x):
    return np.maximum(0, x)

# sigmoid activation function
def sigmod(x):
    return 1 / (1 + np.exp(-x))

# tanh activation function
def tanh(x):
    return np.tanh(x)

# plot the activation function for x in [-10, 10]
x = np.linspace(-10, 10, 100)
#x = np.linspace(-5, 5, 200)
plt.plot(x, relu(x), label='relu')
plt.plot(x, sigmod(x), label='sigmod')
plt.plot(x, tanh(x), label='tanh')
plt.legend()
plt.show()
