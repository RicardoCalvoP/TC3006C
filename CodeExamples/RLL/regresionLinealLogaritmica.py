import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readDataFile(filePath="CodeExamples/RLL/ex2data1.txt"):
    """
    Porper function
    Reads the data from a file and returns two numpy arrays:
    - x_vals: a 2D array with a column of ones for the intercept term
    - y_vals: a 1D array with the target values.

    """
    data = np.loadtxt(filePath, delimiter=",")
    X = data[:, 0:2]   # columnas x1 y x2
    y = data[:, 2]     # columna label

    return X, y

def h0(z):
    return 1/(1 + np.exp(-z))

def gradiente_Descendente(x, y, theta, alpha, iteraciones=1500):
    J = []
    m = len(y)


    for _ in range(iteraciones):
        temp0 = 0
        temp1 = 0
        temp2 = 0
        for i in range(m):
            z = theta[0] + theta[1]*x[i,0] + theta[2]*x[i,1]

            temp0 = temp0 + (h0(z) - y[i])
            temp1 = temp1 + (h0(z) - y[i]) * x[i,0]
            temp2 = temp2 + (h0(z) - y[i]) * x[i,1]

        theta0 = theta[0] - (alpha/m) * temp0
        theta1 = theta[1] - (alpha/m) * temp1
        theta2 = theta[2] - (alpha/m) * temp2

        theta[0] = theta0
        theta[1] = theta1
        theta[2] = theta2


    return theta, J



def regresion_lineal_logaritmica():
  x, y = readDataFile()
  theta = np.zeros(3)
  # Set the learning rate and number of iterations
  alpha = 0.01
  # Set the number of iterations
  iterations = 1500

  # Perform gradient descent
  theta, J = gradiente_Descendente(x, y, theta, alpha, iterations)
  print("Final theta: ", theta)



if __name__ == "__main__":
    regresion_lineal_logaritmica()
