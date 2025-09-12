# bam.py
import numpy as np

def train_bam(A, B, normalize=True):
    """
    A: P×N  (entradas)
    B: P×M  (salidas)
    W: M×N  (B^T A)
    """
    A = np.asarray(A, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    W = (B.T @ A).astype(np.float32)
    if normalize:
        W /= A.shape[0]  # divide por P
    return W  # M×N

def _sgn(u, prev=None):
    out = np.empty_like(u, dtype=np.int8)
    out[u > 0] = 1
    out[u < 0] = -1
    tie = (u == 0)
    if prev is None:
        out[tie] = 1  # o -1; si quieres, cambia
    else:
        out[tie] = prev[tie]  # conserva bit previo en empate
    return out

def recall_bam(W, x0, max_iter=50):
    """
    W: M×N
    x0: N,
    Itera: y = sgn(W x), x = sgn(W^T y)
    Devuelve (x_final, y_final)
    """
    W = np.asarray(W, dtype=np.float32)
    x = np.asarray(x0, dtype=np.int8).reshape(-1)
    y_prev = None
    for _ in range(max_iter):
        y = _sgn(W @ x, prev=y_prev)          # M
        x_new = _sgn(W.T @ y, prev=x)         # N
        if np.array_equal(x_new, x) and (y_prev is not None) and np.array_equal(y, y_prev):
            break
        x, y_prev = x_new, y
    return x, y
