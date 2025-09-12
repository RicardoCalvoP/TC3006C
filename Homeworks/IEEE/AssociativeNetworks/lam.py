import numpy as np

def to_bipolar(X):
    X = np.asarray(X)
    # Si ya viene en {-1,1}, se respeta. Si viene en {0,1}, mapear 0->-1.
    if np.isin(X, [-1, 1]).all():
        return X.astype(int)
    return np.where(X == 0, -1, 1).astype(int)

def train_lam(As, Bs, normalize=True):
    A = to_bipolar(As)  # P×N
    B = to_bipolar(Bs)  # P×M
    W = B.T @ A         # M×N
    if normalize:
        W = W / A.shape[0]  # opcional: 1/P
    return W

def recall_b(W, a):
    a = to_bipolar(a)           # N
    y = W @ a                   # M
    return np.where(y >= 0, 1, -1)

def recall_a(W, b):
    b = to_bipolar(b)           # M
    x = W.T @ b                 # N
    return np.where(x >= 0, 1, -1)
