# Ricardo Calvo Perez- A01028889
# 08/2025

import numpy as np
import matplotlib.pyplot as plt

def readDataFile(filePath="Homeworks\MathFundamentals\Py2_LogisticLinearRegression\ex2data2.txt"):
    """
    Porper function
    Reads the data from a file and returns two numpy arrays:
    - x_vals: a 2D array with a column of ones for the intercept term
    - y_vals: a 1D array with the target values.

    """
    # Get data from the file divided by ','
    data = np.loadtxt(filePath, delimiter=",")
    # Get all data for the first two columns
    X = data[:, 0:2]
    # Get results from the last column
    y = data[:, 2]

    return X, y

def graficaDatos(X, y, theta):

    admitted = y == 1
    not_admitted = y == 0
    plt.scatter(X[admitted, 0], X[admitted, 1], marker='x', color='green', label='Aceptado')
    plt.scatter(X[not_admitted, 0], X[not_admitted, 1], marker='o', color='red', label='Rechazado')

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))

    for i, u_val in enumerate(u):
        for j, v_val in enumerate(v):
            mapped_feature = mapeoCaracteristicas(np.array([[u_val, v_val]]))
            z[i, j] = mapped_feature @ theta

    z = z.T
    plt.contour(u, v, z, levels=[0], linewidths=2, colors='blue', label='Frontera de Decisión')

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Resultados de Chips')
    plt.legend()
    plt.show()

def mapeoCaracteristicas(X):
   # Separate data to different x
   # x1 is our first column
   x1 = X[:, 0:1]
   # x2 is our second column
   x2 = X[:, 1:2]

   # Initialize the new feature matrix with a column of ones for the bias term
   out = np.ones(x1.shape)

   # polynomial degree = stop range - 1
   for i in range (1,7):
      for j in range(i+1):
        # Set i polynomial degree
        new_feature = (x1**(i - j)) * (x2**j)
        out = np.hstack((out, new_feature))

   return out



def sigmoidal(z):
   return 1/(1 + np.exp(-z))


def funcionCostoReg(theta, X, y, lmbda = 1):

   m = len(y)
   # Calculate the linear combination
   z = X @ theta
   # Calculate the hypothesis (predicted probabilities)
   h = sigmoidal(z)
   # Cost regression function
   Jq = (1/m) * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))
   # Regularization parameter
   reg = (lmbda / (2 * m)) * np.sum(theta[1:]**2)
   # Total regularized cost
   J = Jq + reg
   # Gradient cost
   Q0 = (1/m) * X.T @ (h - y)
   # Add the regularization term to the gradient.
   # The regularization term for grad is (lambda / m) * theta
   Q1 = (lmbda / m) * theta
   # Don't regularize the gradient for the first parameter (theta[0])
   Q1[0] = 0
   # Calculate gradient by the sum of Q when j = 0 & Q when j >= 1
   grad = Q0 + Q1
   # Flatten the gradient vector to return a 1D array
   grad = grad.flatten()

   return J, grad

def aprende(theta, X, y, iteraciones, lmbda = 1):
   """
   """
   alpha = 0.01
   for _ in range(iteraciones):
      J , grad = funcionCostoReg(theta, X, y, lmbda)
      theta = theta - alpha * grad

   return theta
def predice(theta, X):
   z = X @ theta

   # Calculate the predicted probabilities (h)
   h = sigmoidal(z)

   # Convert probabilities to a binary prediction (0 or 1)
   # The threshold is 0.5.
   p = (h >= 0.5).astype(int)

   return p


def main():
  # Global values
  # number of iterations for the examples
  iterations = 15000
  # lambda value
  lmbda = 1
  # Read data from the file
  X, y = readDataFile()
  # Increased X vector
  X_mapped = mapeoCaracteristicas(X)
  # Initialize theta in 0 from the increased X vector size
  theta = np.zeros((X_mapped.shape[1]))

  # Function example to get cost = 0.693 as exercise says
  J, _ = funcionCostoReg(theta, X_mapped, y, lmbda)
  print(J)

  # Second Example to get accuracy to 83.050847%
  theta = aprende(theta, X_mapped, y, iterations)
  p = predice(theta, X_mapped)

  acc = 0
  for prediction, y_real in zip(p, y):
     if prediction == y_real:
        acc = acc + 1

  acc = (acc /len(y)) * 100

  print(f"{acc:.6f}%")

  # Graph given data.
  graficaDatos(X, y, theta)

  lmbda_tests = [1, 0, 100]

  for l in lmbda_tests:
      theta = aprende(theta, X_mapped, y, iterations, l)
      p = predice(theta, X_mapped)

      acc = 0
      for prediction, y_real in zip(p, y):
        if prediction == y_real:
            acc = acc + 1

      acc = (acc /len(y)) * 100

      print(f"{acc:.6f}%")

      # Graph given data.
      graficaDatos(X, y, theta)


if __name__ == "__main__":
  main()