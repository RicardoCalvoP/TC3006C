# Ricardo Calvo Perez - A01028889

import numpy as np

def readDataFile(filePath="Homeworks/MathFundamentals/Py3_BackPropagation/digitos.txt"):
    X, y = [], []
    with open(filePath) as file:
        for line in file:
            text_list = line.strip().split()
            if not text_list:
                continue
            X.append([float(v) for v in text_list[:-1]])  # 400 pixel values
            y.append(int(float(text_list[-1])))           # label

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    return X, y


def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y, iterations = 1500, alpha = 0.9):

  W1, b1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
  W2, b2 = randInicializacionPesos(hidden_layer_size, num_labels)

  m = X.shape[0]
  # one-hot targets
  Y = np.zeros((m, num_labels), dtype=float)
  Y[np.arange(m), y.astype(int) - 1] = 1.0

  J = []
  eps = 1e-12  # numerical stability

  for iteratin in range(iterations):
    net1 = X.dot(W1.T) + b1
    O1 = sigmoidalGradiente(net1)
    net2 =  O1.dot(W2.T) + b2
    O2 = sigmoidalGradiente(net2)

    P = np.clip(O2, eps, 1.0 - eps)
    J_temp = -(1.0/m) * np.sum(Y*np.log(P) + (1.0 - Y)*np.log(1.0 - P))
    J.append(J_temp)

    # ---- backprop ----
    delta3 = O2 - Y                                 # (m, K)
    delta2 = (delta3 @ W2) * (O1 * (1.0 - O1))      # (m, H)

    dW2 = (delta3.T @ O1) / m                       # (K, H)
    db2 = delta3.mean(axis=0)                       # (K,)
    dW1 = (delta2.T @ X) / m                        # (H, D)
    db1 = delta2.mean(axis=0)                       # (H,)

    # ---- gradient descent ----
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

  return W1, b1, W2, b2, J



def sigmoidalGradiente(z):
  return 1/(1+ np.exp(-z))

def randInicializacionPesos(L_in, L_out):

  eps = 0.12
  W = np.random.uniform(-eps, eps, size = (L_out, L_in))
  b = np.random.uniform(-eps, eps, size = (L_out,))

  return W, b

def prediceRNYaEntrenada(X, W1, b1, W2, b2):

  net1 = X.dot(W1.T) + b1
  O1 = sigmoidalGradiente(net1)

  net2 = O1.dot(W2.T) + b2
  O2 = sigmoidalGradiente(net2)
  return np.argmax(O2, axis=1) + 1  # 1..10


def main():
  X, y = readDataFile()

  input_layer_size = 400 # Set in instructions
  hidden_layer_size = 25 # Set in instructions
  num_labels = 10 # Set in instructions

  W1, b1, W2, b2, J = entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y)
  O2 = prediceRNYaEntrenada(X, W1, b1, W2, b2)
  acc = np.mean(O2 == y)
  print("J final: ", J[-1])      # costo de la última época
  print("O2: ", O2)      # costo de la última época
  print(f"acc: {acc*100}%")

  # prediceRNYaEntrenada(X, W1, b1, W2, b2)


if __name__ == "__main__":
  main()