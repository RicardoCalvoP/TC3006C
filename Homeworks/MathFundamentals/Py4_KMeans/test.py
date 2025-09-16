import numpy as np
import matplotlib.pyplot as plt

def readData(file):
    with open(file, 'r', encoding='utf-8') as f:
        x = []
        y = []
        for line in f:
            line = line.strip()  # Eliminar espacios en blanco al inicio y al final
            if not line:  # Ignorar líneas vacías
                continue
            try:
                a, b = line.split()  # Intentar dividir la línea en dos valores
                x.append(float(a))
                y.append(float(b))
            except ValueError:
                print(f"Línea ignorada: {line}")  # Mostrar un mensaje si la línea no es válida

    return np.array(x), np.array(y)

def graficaClusters(x, y, labels, centroids):
    """
    Grafica los datos con colores distintos para cada cluster y los centroides.
    
    Args:
        x (np.array): Coordenadas X de los datos.
        y (np.array): Coordenadas Y de los datos.
        labels (np.array): Etiquetas de los clusters asignadas a cada punto.
        centroids (np.array): Coordenadas de los centroides.
    """
    plt.figure(figsize=(8, 6))
    data = np.column_stack((x, y))
    k = len(np.unique(labels))  # Número de clusters
    colors = plt.cm.get_cmap('tab10', k)  # Generar una paleta de colores

    for i in range(k):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', color=colors(i))

    # Graficar los centroides
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroides')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Clusters generados por k-means')
    plt.grid(True)
    plt.show()


def kmeans(x, y, k=3, max_iters=100):
    data = np.column_stack((x, y))
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels


def main():
    x, y = readData('ex7data2.txt')
    centroids, labels = kmeans(x, y, k=30)
    graficaClusters(x, y, labels, centroids)

main()