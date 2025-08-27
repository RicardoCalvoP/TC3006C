# Ricardo Calvo Perez - A01028889
# 08/2025
import numpy as np

def output(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5):
    """
    Computes the output of a simple perceptron model based on the input features and weights.
    y = f(Sum(w_i * x_i) + (\theta * W0)) where i ranges from 1 to n.
    """
    y = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5
    if y >= 0:
        return 1
    else:
        return -1

def get_error(y, y_hat):
    """
    Computes the error between the predicted output and the actual output.
    """
    return y - y_hat

def calculate_weights(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, y, n=0.1):
    """
    Updates the weights based on the input features and the error.
    The weights are updated using the perceptron learning rule.
    """
    y_hat = output(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5)
    error = get_error(y, y_hat)

    # Update weights
    w1 += (n * error * x1)
    w2 += (n * error * x2)
    w3 += (n * error * x3)
    w4 += (n * error * x4)
    w5 += (n * error * x5)

    return w1, w2, w3, w4, w5

def verify_weights(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5):
    """
    Verifies the weights by checking if they are within a reasonable range.
    """
    # Make weights into a array for easier manipulation
    weights = np.array([w1, w2, w3, w4, w5])
    # Multiply for x1, x2, x3, x4, x5
    inputs = np.array([x1, x2, x3, x4, x5])
    # Calculate the output
    y_hat = np.dot(weights, inputs)
    # Check if the output matches the expected value
    if y_hat >= 0:
        return 1
    else:
        return 0
def main():
    w1 = np.random.rand()
    w2 = np.random.rand()
    w3 = np.random.rand()
    w4 = np.random.rand()
    w5 = np.random.rand()

    x1 = [0.8, -0.5, 0.9, 0.0, 0.7, -0.1, 0.6, 0.2, 1.0, -0.2, 0.5, 0.1]
    x2 = [0.2, 0.7, 0.1, 0.5, 0.3, 0.8, 0.1, 0.4, 0.0, 0.6, 0.2, 0.3]
    x3 = [0.9, 0.3, 0.8, 0.4, 0.7, 0.1, 0.6, 0.2, 1.0, 0.0, 0.7, 0.4]
    x4 = [0.7, 0.2, 0.9, 0.3, 0.8, 0.1, 0.7, 0.4, 1.0, 0.0, 0.6, 0.5]
    x5 = [0.5, 1.1, 0.6, 0.8, 0.4, 1.5, 0.5, 0.9, 0.3, 1.2, 0.3, 0.7]
    y  = [1,   0,   1,   0,   1,   0,   1,   0,   1,   0,   1,   0]


    print(f"Initial weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}")

    for i in range(len(x1)):
        y_hat = output(x1[i], x2[i], x3[i], x4[i], x5[i], w1, w2, w3, w4, w5)
        error = get_error(y[i], y_hat)
        if error != 0:
            w1, w2, w3, w4, w5 = calculate_weights(x1[i], x2[i], x3[i], x4[i], x5[i], w1, w2, w3, w4, w5, y[i])
            i = 0  # Reset index to re-evaluate all inputs after weight update

    print(f"Final weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}")

    x1_test = [0.9, -0.3, 0.8]
    x2_test = [0.0, 0.5, 0.1]
    x3_test = [0.9, 0.1, 0.8]
    x4_test = [0.9, 0.2, 0.8]
    x5_test = [0.6, 1.0, 0.4]
    y_test = [1, 0, 1]

    for i in range(len(x1_test)):
        y_hat = verify_weights(x1_test[i], x2_test[i], x3_test[i], x4_test[i], x5_test[i], w1, w2, w3, w4, w5)
        print(f"Predicted output: {y_hat}, Expected output: {y_test[i]}")

if __name__ == "__main__":
    main()