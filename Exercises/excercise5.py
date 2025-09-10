import numpy as np
import os

def stair_act_func(y):
  y_pred =  1 if y >= 0 else -1
  return y_pred

def output(W0, W1, W2, W3, x0, x1, x2, x3):
  """"""
  y1 = W0*x0 + W1*x1 + W2*x2 + W3*x3
  y = stair_act_func(y1)
  print (f"$y = ({W0:.2f})({x0:.2f}) + ({W1:.2f})({x1:.2f}) + ({W2:.2f})({x2:.2f}) + ({W3:.2f})({x3:.2f}) = {y1:.2f} -> f({y1:.2f}) = {y:.2f}$\n")
  return y

def update_weights(W0, W1, W2, W3, x0, x1, x2, x3, n, e):
 oldW0, oldW1, oldW2, oldW3 = W0, W1, W2, W3  # para mostrar antes→después

 print("**Recalculate weights**")
 print(f"- $W_0 = {oldW0:g} + ({n:g} *({e:g}) * {x0:g}) = {oldW0 + n*e*x0:g}$")
 print(f"- $W_1 = {oldW1:g} + ({n:g} *({e:g}) * {x1:g}) = {oldW1 + n*e*x1:g}$")
 print(f"- $W_2 = {oldW2:g} + ({n:g} *({e:g}) * {x2:g}) = {oldW2 + n*e*x2:g}$")
 print(f"- $W_3 = {oldW3:g} + ({n:g} *({e:g}) * {x3:g}) = {oldW3 + n*e*x3:g}$")
 # actualizaciones (tu lógica original)
 W0 = W0 + (n * (e) * x0)
 W1 = W1 + (n * (e) * x1)
 W2 = W2 + (n * (e) * x2)
 W3 = W3 + (n * (e) * x3)

 return W0, W1, W2, W3

def main():
  X = np.array([
    [1.0,  1.0,  1.0,  1.0,  1.0],
    [1.0,  1.0,  1.0, -1.0,  1.0],
    [1.0,  1.0, -1.0,  1.0,  1.0],
    [1.0,  1.0, -1.0, -1.0, -1.0],
    [1.0, -1.0,  1.0,  1.0,  1.0],
    [1.0, -1.0,  1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0,  1.0, -1.0],
    [1.0, -1.0, -1.0, -1.0, -1.0],
], dtype=float)

  W0 = 0.5
  W1 = 0.5
  W2 = 0.2
  W3 = 0.8

  n = 0.5

  while True:
        i = 0
        hubo_cambio = False

        # Recorre desde la primera línea. Si hay error y se actualiza,
        # reinicia desde la línea 0 con los NUEVOS pesos.
        while i < len(X):
            print(f"**Line {i+1}**\n")
            x0, x1, x2, x3, sd = X[i]
            yhat = output(W0, W1, W2, W3, x0, x1, x2, x3)
            e = sd - yhat  # perceptrón: error en { -2, 0, +2 :.2f}
            print(f"$e = {sd} - ({yhat}) = {e}$\n")
            if e != 0:
                print(f"**Error in line {i+1}**\n")
                W0, W1, W2, W3 = update_weights(W0, W1, W2, W3, x0, x1, x2, x3, n, e)

                hubo_cambio = True
                i = 0     # reinicia desde la PRIMERA línea
                print(f"\nNew wights: W0={W0:.2f}, W1={W1:.2f}, W2={W2:.2f}, W3={W3:.2f}\n")
                continue
            i += 1

        if not hubo_cambio:
            print("\nNo errors in any line.")
            print(f"Final weights: W0={W0:.2f}, W1={W1:.2f}, W2={W2:.2f}, W3={W3:.2f}\n")
            break

if __name__ ==  "__main__":
  os.system("cls")
  main()
