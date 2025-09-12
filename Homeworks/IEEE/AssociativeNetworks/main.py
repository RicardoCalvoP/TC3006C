# Ricardo Calvo - A01028889
# Melissa Mireles - A01379736
# 09/2025
import os
import numpy as np

import utils
from hopfield import hopfield_network
from lam import train_lam, recall_b
from bam import train_bam, recall_bam


def main():
  utils.process_folder("figures", "patterns_txt")
  patterns_dict = utils.fill_patterns()
  patterns = list(patterns_dict.values())

  A = np.stack(list(patterns_dict.values()))
  B = A

  # Define pares (entrada -> salida)
  pairs = [("star", "lightning"),
          ("oval",   "rectangle"),
          ("triangle",   "cross"),
          ("square",   "rectangle"),
          ]

  bam_A = np.stack([patterns_dict[a] for a, b in pairs])  # P×4096
  bam_B = np.stack([patterns_dict[b] for a, b in pairs])  # P×4096
  for name, pattern in patterns_dict.items():
    test_pattern = pattern.copy()                # o una versión con ruido
    recovered = hopfield_network(patterns, test_pattern)
    # utils.show_pattern(pattern, shape=(64, 64), title=name)    # ahora sí se ve la figura

    W = train_lam(A, B, normalize=True)
    y = recall_b(W, patterns_dict[name])                        # 4096 en {-1,1}
    # utils.show_pattern(y, shape=(64,64), title=f"recall {name}")

  W = train_bam(bam_A, bam_B, normalize=True)

  for a, b in pairs:
    x0 = patterns_dict[a]
    x_rec, y_rec = recall_bam(W, x0)
    utils.show_pattern(x_rec, shape=(64, 64), title=f"x≈{a}")
    utils.show_pattern(y_rec, shape=(64, 64), title=f"y≈{b}")


if __name__ == "__main__":
  os.system("cls")
  np.set_printoptions(threshold=np.inf)
  main()
  print("Succes!")