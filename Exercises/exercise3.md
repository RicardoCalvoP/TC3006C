## Exercise 3

Build a system that automatically place the water bottles in a green conveyor belt and soda cans in a blue conveyor belt. Design a neural network that classify between water bottles and soda cans.

Water Bottles

- b1 = [5, 2]
- b2 = [5, 4.5]
- b3 = [6, 6]
- b4 = [8, -1]

Cans

- c1 = [1, 2]
- c2 = [1, -5]
- c3 = [-1, 0]
- c4 = [-2, -2]

1. Draw the patterns in a chart

   ![alt text](/Exercises/imgs/exercise1/image.png)

   Where Blue are Cans & Green are Bottle of Water

1. Draw the decision boundary

   ![alt text](/Exercises/imgs/exercise1/image-1.png)

1. Get two values from the line

   > [!WARNING]
   > Chosen coordinates

   - m1 [2, 4]
   - m2 [4, 0]

1. Get Slope

   $y=mx+b$

   $m=(0-4)/(4-2) = -4/2 = -2$

1. Get value of `b`

   $y=-2x+b$

   $2=-2(4)+b$

   $2+8=+b$

   $b=10$

1. Get the `w` vector and scale `b`

   $y=-2x+b$

   $-2x-y+10=0$
