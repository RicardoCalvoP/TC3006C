import numpy as np
import matplotlib.pyplot as plt
import os
_and = np.array([[1.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 1.0, 0.0],
                 [1.0, 1.0, 0.0, 0.0],
                 [1.0, 1.0, 1.0, 1.0]])

_or = np.array ([[1.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0]])


_xor = np.array ([[1.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0, 0.0]])

dataset = _and

w13 = np.random.normal(0, 1)
w23 = np.random.normal(0, 1)
b3 = np.random.normal(0, 1)
c3 = [w13, w23, b3]

w14 = np.random.normal(0, 1)
w24 = np.random.normal(0, 1)
b4 = np.random.normal(0, 1)
c4 = [w14, w24, b4]

w35 = np.random.normal(0, 1)
w45 = np.random.normal(0, 1)
b5 = np.random.normal(0, 1)
c5 = [w35, w45, b5]


alpha = 0.01
error = 0
def ON(net):
  return 1 / (1 + np.exp(-net))

for iteration in range(10000):
  print("___________________")
  for i in range(len(dataset)):
    x1 = dataset[i][1]
    x2 = dataset[i][2]
    y = dataset[i][3]
    print('|',x1,'|', x2,'|', y, '|')
    net3 = b3 + w13 * x1 + w23 * x2
    O3 = ON(net3)
    net4 = b4 + w14 * x1 + w24 * x2
    O4 = ON(net4)
    net5 = b5 + w35 * O3 + w45 * O4
    O5 = ON(net5)

    # C5
    temp_w35 = -(y-O5) * O5 * (1 - O5) * O3
    temp_w45 = -(y-O5) * O5 * (1 - O5) * O4
    temp_wb5 = -(y-O5) * O5 * (1 - O5)

    # C3
    temp_w13 = -(y-O5) * O5 * (1 - O5) * w35 * O3 * (1 - O3) * x1
    temp_w23 = -(y-O5) * O5 * (1 - O5) * w35 * O3 * (1 - O3) * x2
    temp_wb3 = -(y-O5) * O5 * (1 - O5) * w35 * O3 * (1 - O3)

    # C4
    temp_w14 = -(y-O5) * O5 * (1 - O5) * w45 * O4 * (1 - O4) * x1
    temp_w24 = -(y-O5) * O5 * (1 - O5) * w45 * O4 * (1 - O4) * x2
    temp_wb4 = -(y-O5) * O5 * (1 - O5) * w45 * O4 * (1 - O4)

    w35 = w35 - alpha * temp_w35
    w45 = w45 - alpha * temp_w45
    b5 = b5 - alpha * temp_wb5

    w13 = w13 - alpha * temp_w13
    w23 = w23 - alpha * temp_w23
    b3 = b3 - alpha * temp_wb3

    w14 = w14 - alpha * temp_w14
    w24 = w24 - alpha * temp_w24
    b4 = b4 - alpha * temp_wb4

    error = 1/2 * (y - O5)**2
    if (error < 0.001):
      break

print(error)