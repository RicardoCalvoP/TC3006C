import numpy as np
import matplotlib.pyplot as plt

_and = np.array([[1.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 1.0, 0.0],
                 [1.0, 1.0, 0.0, 0.0],
                 [1.0, 1.0, 1.0, 1.0]])

_or = np.array ([[1.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0]])



dataset = _and

w0 = 1.5
w1 = 0.5
w2 = 1.5

alfa = 0.01

def activation_function(y):
    return 1 if y >= 0 else 0
error_list = []
for epoch in range(1000):
    for i in range(len(dataset)):
        x1 = dataset[i][1]
        x2 = dataset[i][2]
        y = dataset[i][3]
        Net = w0 + w1 * x1 + w2 * x2
        output = activation_function(Net)
        error = y - output
        w0 += alfa * error
        w1 += alfa * error * x1
        w2 += alfa * error * x2

    error_list.append(error)


plt.plot(error_list)
plt.show()
print(w0,w1, w2)