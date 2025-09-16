import numpy as np

def main():

    # Pregunta 3 ------------------------------------------
    x = np.array([2, 3, 5, 4, 3, 6, 1], dtype=float)
    y = np.array([4, 6, 8, 4, 2, 3, 1], dtype=float)

    # Matriz de diseño: [1, x, x^2, x^3]
    X = np.column_stack([np.ones_like(x), x, x**2, x**3])

    # Resolver ecuación normal: (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    print("Coeficientes:", theta)
    print("theta_2 =", theta[2])

    # Pregunta 4 ------------------------------------------
    # Centroides
    C1 = np.array([0, 1, 1])
    C2 = np.array([4, 1, 2])

    # Puntos
    points = np.array([
        [2, 3, 5],
        [1, 3, 2],
        [6, 2, 4],
        [-1, 1, 3]
    ])

    # Calcular distancias al cuadrado
    dist_C1 = np.sum((points - C1)**2, axis=1)
    dist_C2 = np.sum((points - C2)**2, axis=1)

    # Asignar al cluster más cercano
    clusters = np.where(dist_C1 < dist_C2, 1, 2)

    print(clusters)

    # Matriz de componentes
    C = np.array([
        [0.01, -0.36, -0.29],
        [-0.21, 0.54, 0.73],
        [-0.06, 0.76, 0.23]
    ])

    # Ejemplo estandarizado
    X = np.array([0.92, 1.56, 0.12])

    # Segunda columna de C (segundo componente)
    c2 = C[:, 1]
    print(c2)

    # Coordenada proyectada sobre el segundo componente
    z2 = np.dot(X, c2)

    print("Coordenada respecto al segundo componente:", round(z2, 4))

    # Pregunta 6 y 7
    sig = lambda z: 1 / (1 + np.exp(-z))

    # Datos
    x1 = np.array([0.8, 0.3, 0.2])  # X(1)
    W1 = np.array([[0.2, 0.3, 0.2],
                [0.3, 0.1, 0.1],
                [0.4, 0.5, 0.3]])
    b1 = np.array([0.1, 0.4, 0.7])

    W2 = np.array([[-0.1],
                [ 0.6],
                [-0.5]])
    b2 = np.array([0.2])

    y = 1.0  # y(1)

    # Forward
    z1 = x1 @ W1 + b1
    a1 = sig(z1)

    z2 = a1 @ W2 + b2
    y_hat = sig(z2).item()

    # Gradiente para BCE con sigmoide en la salida
    delta2 = y_hat - y
    grad_W2 = a1[:, None] * delta2
    grad_w47 = grad_W2[0, 0]

    print(f"y_hat = {y_hat:.4f}")
    print(f"dJ/dw47 = {grad_w47:.4f}")





if __name__ == "__main__":
  main()