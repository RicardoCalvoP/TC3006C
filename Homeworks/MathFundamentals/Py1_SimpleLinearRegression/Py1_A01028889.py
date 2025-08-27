# Ricardo Calvo Perez- A01028889
# 08/2025

import numpy as np
import matplotlib.pyplot as plt

def readDataFile(filePath="Homeworks/MathFundamentals/Py1_SimpleLinearRegression/ex1data1.txt"):
    """
    Porper function
    Reads the data from a file and returns two numpy arrays:
    - x_vals: a 2D array with a column of ones for the intercept term
    - y_vals: a 1D array with the target values.

    """
    x_vals = []
    y_vals = []
    with open(filePath) as file:
        lines = file.readlines()

        # skip first line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            x, y = map(float, parts)
            x_vals.append(x)
            y_vals.append(y)

    # Add a column of ones for the intercept term
    x_vals = np.column_stack([np.ones(len(x_vals)), x_vals])
    # Convert lists to numpy arrays
    y_vals = np.array(y_vals)


    return  x_vals, y_vals

def graficaDatos(x_data, y_data, theta):
    """
    It receives input data and graphs them as points on an (x,y) plane.
    It also receives a theta vector and graphs the line
    resulting from those values over the data.
    """

    # Plot the data points first
    plt.scatter(x_data[:, 1], y_data ,color='blue', label='Data Points')
    # Create the x-axis values for the line
    x_values = np.array([x_data[:, 1].min(), x_data[:, 1].max()])
    # Calculate the y-axis values for the line using the final theta
    y_values = theta[0] + theta[1] * x_values
    # Plot the line
    plt.plot(x_values, y_values, color='red', label='Linear Regression')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Linear Regression with Gradient Descent')
    plt.legend()
    plt.show()


def gradienteDescendente(x, y, theta, alpha, iteraciones = 1500):
  """
  a. Input data already separated into vectors X and y.
  b. Initial vector theta = [q0,q1]. This vector can be initialized to 0 for both values.
  c. Learning rate alpha. It is recommended to test it with a value of alpha = 0.01.
  d. The number of iterations iterations. The number of iterations the algorithm will perform.
    It is recommended to test it with iterations = 1500.
  e. It can be done in batch or online (the results given for testing were obtained with batch,
    but they should be very close with online).
  The function should return the value of the final vector theta.
  """

  # Initialize the cost function history
  J = []
  # Perform gradient descent for the specified number of iterations
  for _ in range(iteraciones):
      # Calculate the error
      error = x.dot(theta) - y
      # Update theta using the gradient descent formula
      # theta = theta - (alpha / m) * (X^T * error)
      theta = theta - (alpha / len(y)) * (x.T.dot(error))
      # Calculate the cost and append it to the history
      J.append(calculaCosto(x, y, theta))
  return theta, J


def calculaCosto(x, y, theta):
    """
    It receives the inputs x and y, and a vector theta and returns the value of the cost function J(q0, q1) that results.

    # Calculate the cost function J
    # J(q0, q1) = (1/2m) * sum((h(xi) - yi)^2)
    """

    et = (x.dot(theta) - y)**2
    return et.sum() / (2 * len(y))

def main():
  # Read data from the file
  x, y = readDataFile()
  # Initialize theta as a vector of zeros
  theta = np.zeros(2)
  # Set the learning rate and number of iterations
  alpha = 0.01
  # Set the number of iterations
  iterations = 1500
  # Perform gradient descent
  theta, J =  gradienteDescendente(x, y, theta, alpha, iterations)
  print("Final theta: ", theta)
  print("Final cost J: ", J[-1])
  # Plot the results
  graficaDatos(x,  y, theta)
  plt.plot(J, label='Cost Function J')
  plt.show()

if __name__ == "__main__":
  main()