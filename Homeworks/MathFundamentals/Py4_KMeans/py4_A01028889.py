# Ricardo Calvo - A01028889
# 09/2025

import numpy as np
import matplotlib.pyplot as plt

def read_image_txt_planar(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    H, W, C = map(int, lines[0].split())
    assert C == 3, "Se esperan 3 canales RGB"
    vals = np.array(list(map(float, lines[1:])), dtype=np.float32)
    N = H * W
    assert vals.size == 3 * N, f"Se esperaban {3*N} valores, recibidos {vals.size}"

    # Split values into R, G, B channels
    R = vals[0:N].reshape(H, W)
    G = vals[N:2*N].reshape(H, W)
    B = vals[2*N:3*N].reshape(H, W)

    # Stack channels into image array
    A = np.stack([R, G, B], axis=2)
    A = np.clip(A, 0, 255).astype(np.uint8)

    # Flatten into (m,3) for K-means
    X = A.reshape(H*W, 3)
    return X, (H, W, 3), A

def findClosestCentroids(X, initial_centroids):
    # Compute distances and assign each point to nearest centroid
    dists = np.linalg.norm(X[:, None, :] - initial_centroids[None, :, :], axis=2)
    idx = np.argmin(dists, axis=1)
    return idx

def computeCentroids(X, idx, K):
    # Compute mean of points for each cluster
    d = X.shape[1]
    newC = np.zeros((K, d), dtype=np.float32)
    for k in range(K):
        mask = (idx == k)
        if np.any(mask):
            newC[k] = X[mask].mean(axis=0)
        else:
            # Handle empty cluster by reinitializing randomly
            newC[k] = X[np.random.randint(0, X.shape[0])]
    return newC

def runkMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    X = X.astype(np.float32)
    K = initial_centroids.shape[0]
    centroids = initial_centroids.astype(np.float32)
    history = [centroids.copy()]           # C(0)

    for _ in range(max_iters):
        # asign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        idx = np.argmin(dists, axis=1)
        # update
        newC = np.zeros_like(centroids)
        for k in range(K):
            m = (idx == k)
            newC[k] = X[m].mean(0) if np.any(m) else X[np.random.randint(0, X.shape[0])]
        history.append(newC.copy())
        if np.allclose(newC, centroids, atol=1e-5):
            centroids = newC
            break
        centroids = newC

    return centroids, idx, history

def kMeansInitCentroids(X, K, seed=42):
    # Randomly choose K data points as initial centroids
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    return X[rng.permutation(m)[:K]].astype(np.float32)

def reconstruct_image(idx, centroids, dims, flipud=True):
    # Map each pixel to its centroid color
    H, W, C = dims
    Xc = centroids[idx]
    A = np.clip(Xc, 0, 255).astype(np.uint8).reshape(H, W, C)
    if flipud:
        A = np.flipud(A)
    return A

def show_image(A, title=None):
    # Display image with matplotlib
    plt.imshow(A); plt.axis('off')
    if title: plt.title(title)
    plt.show()

def pca2(X):
    # Center data and compute top-2 PCA directions
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:2].T; mu = X.mean(0)
    return mu, P

def plot_clusters(X, idx, history, title=None):
    # Project data and centroids to 2D using PCA
    mu, P = pca2(X.astype(np.float32))
    X2 = (X - mu) @ P
    H = np.stack(history)                  # t×K×d (centroids at each iteration)
    K = H.shape[1]
    C2_all = ((H - mu) @ P)                # trajectories in 2D

    plt.figure(figsize=(7,5))

    # Use final centroid RGB values as cluster colors
    centroids_final = H[-1]
    colors = np.clip(centroids_final / 255.0, 0, 1)

    # Plot points colored by their final centroid
    for k in range(K):
        pts = X2[idx == k]
        plt.scatter(pts[:,0], pts[:,1], s=14, color=colors[k], alpha=0.7, label=f'Cluster {k+1}')

    # Plot centroid trajectories in black
    for k in range(K):
        traj = C2_all[:,k,:]
        plt.plot(traj[:,0], traj[:,1], '-k', linewidth=1.5)
        plt.plot(traj[:,0], traj[:,1], 'kx', markersize=6)

    # Highlight final centroids with their cluster color
    C_final_2D = C2_all[-1,:,:]
    for k in range(K):
        plt.scatter(C_final_2D[k,0], C_final_2D[k,1],
                    c=colors[k].reshape(1,-1), s=200, marker='X', edgecolors='k')

    plt.grid(True)
    plt.title(title or f'Iteration number {len(history)-1}')
    plt.tight_layout()
    plt.show()

def main():
    X, dims, A = read_image_txt_planar("bird_small.txt")

    A_show = np.flipud(A)
    # Show the original image to compare results
    show_image(A_show, "Original")
    # Test values of k clusters to see the kmean behaviour of our algorithm
    Ks = [2, 4, 8, 16, 32, 64, 512]
    for K in Ks:
        initC = kMeansInitCentroids(X.astype(np.float32), K, seed=42)
        C, idx, history = runkMeans(X.astype(np.float32), initC, max_iters=20, plot_progress=True)
        plot_clusters(X, idx, history)
        A_comp = reconstruct_image(idx, C, dims, flipud=True)
        show_image(A_comp, f"Comprimida K={K}")


if __name__ == "__main__":
  main()
