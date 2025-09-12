import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def image_to_pattern(path, threshold=200):
    """
    Convierte una imagen a una matriz {-1,1} sin cambiar su tamaño original.
    threshold: valor de corte (0-255) para binarizar.
    """
    img = Image.open(path).convert("L")  # escala de grises
    arr = np.array(img)
    pattern = np.where(arr < threshold, 1, -1)
    return pattern  # matriz del mismo tamaño que la imagen

def process_folder(input_dir="figures", output_dir="patterns_txt"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png",".jpg",".jpeg",".bmp")):
            path = os.path.join(input_dir, fname)
            pattern = image_to_pattern(path)
            out_name = os.path.splitext(fname)[0] + ".txt"
            out_path = os.path.join(output_dir, out_name)
            # Guardar como matriz (una fila por línea)
            np.savetxt(out_path, pattern, fmt="%d")
            print(f"Guardado: {out_path}, tamaño {pattern.shape}")

def fill_patterns(patterns_dir="patterns_txt"):
    patterns = {}
    for fname in sorted(os.listdir(patterns_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() != ".txt":
            continue
        path = os.path.join(patterns_dir, fname)
        values = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    values.extend(int(x) for x in line.split())
        patterns[base] = np.asarray(values, dtype=int)  # vector 1D
    return patterns

def show_pattern(pattern, shape=None, title=None, save_path=None):
    """
    Muestra un patrón donde -1=negro y 1=blanco.
    pattern: vector 1D o matriz 2D de -1/1
    shape: (H,W) si pattern viene plano. Si ya es 2D, omítelo.
    """
    arr = np.asarray(pattern, dtype=int)
    if arr.ndim == 1:
        if shape is None:
            raise ValueError("Proporciona shape=(alto, ancho) para vector 1D.")
        if arr.size != shape[0]*shape[1]:
            raise ValueError(f"Tamaño {arr.size} != {shape[0]*shape[1]}.")
        arr = arr.reshape(shape)
    elif arr.ndim != 2:
        raise ValueError("pattern debe ser 1D o 2D.")

    img = (arr + 1) // 2  # -1->0, 1->1
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()