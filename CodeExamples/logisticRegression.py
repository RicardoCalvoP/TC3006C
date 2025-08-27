import numpy as np
import math
import random

def sigmoid(z):
    """
    Computes the sigmoid function for a given input z.
    """
    return 1 / (1 + math.exp(-z))

def gradient(sampleList, weights):
    sumElements = 0.0

    for x, y in zip(sampleList, weights):
        sumElements += (x*y)

    return sigmoid(sumElements)

def stochasticGradientAscent(trainingLists, trainingLabels, featureNumber ,iterarions=150):
  #Get the number of training samples
  sampleNumber=len(trainingLists)

    #Create a list of N features (featureNumber) for saving optimal weights (1.0 as initial value)
  weights=[1.0] * featureNumber
  #Iterate a fixed number of times for getting optimal weights
  for x in range(iterarions):
    #Get the index number of training samples
    sampleIndex = list(range(sampleNumber))
    #For each training sample do the following
    for y in range(sampleNumber):
      """
      Alpha is the learning rate and controls how much the coefficients (and therefore the model)
      changes or learns each time it is updated.
      Alpha decreases as the number of iterations increases, but it never reaches 0
      """
      alpha=4/(1.0+x+y)+0.01
      #Randomly obtain an index of one of training samples
      """
      Here, you’re randomly selecting each instance to use in updating the weights.
      This will reduce the small periodic variations that can be present if we analyze
      everything sequentially
      """
      randIndex = int(random.uniform(0,len(sampleIndex)))
      #Obtain the gradient from the current training sample and weights
      sampleGradient=gradient(trainingLists[randIndex],weights)
      #Check the error rate
      error=trainingLabels[randIndex]-sampleGradient
      """
      we are calculating the error between the actual class and the predicted class and
      then moving in the direction of that error (CURRENT TRAINING PROCESS)
      """
      temp=[]
      for index in range(featureNumber):
       temp.append(alpha*(error*trainingLists[randIndex][index]))

      for z in range(featureNumber):
        weights[z]= weights[z] + temp[z]

      del(sampleIndex[randIndex])
  return weights

def classifyList(testList, weights):
  sumElements=0
  #Multiply all features and optimized weights
  for x,y in zip(testList,weights):
    sumElements=sumElements+(x*y)
    #Obtain the sigmoid output which will tell us the class a test vector belongs
    probability = sigmoid(sumElements)
  if probability > 0.5:
    return 1.0
  else:
    return 0.

def logisticRegression():
  base_path = "./Logistic_Regression/"

  num_data = 100

  feature1=list(np.random.normal(0, 0.2, num_data))
  feature2=list(np.random.normal(0, 0.2, num_data))
  feature3=list(np.random.normal(0, 0.2, num_data))
  feature4=list(np.random.normal(0, 0.2, num_data))
  feature5=list(np.random.normal(0, 0.2, num_data))
  label=list(np.random.choice([1.0, 0.0], size=50, p=[0.5, 0.5]))

  with open(base_path + "regressionTraining.txt","w") as file:
    for f1,f2,f3,f4,f5,l in zip(feature1,feature2,feature3,feature4,feature5,label):
      file.write(str(f1)+","+str(f2)+","+str(f3)+","+str(f4)+","+str(f5)+","+str(l)+"\n")

  feature1=list(np.random.normal(0, 0.2, 50))
  feature2=list(np.random.normal(0, 0.2, 50))
  feature3=list(np.random.normal(0, 0.2, 50))
  feature4=list(np.random.normal(0, 0.2, 50))
  feature5=list(np.random.normal(0, 0.2, 50))
  label=list(np.random.choice([1.0, 0.0], size=50, p=[0.5, 0.5]))

  with open(base_path + "regressionTest.txt","w") as file:
    for f1,f2,f3,f4,f5,l in zip(feature1,feature2,feature3,feature4,feature5,label):
      file.write(str(f1)+","+str(f2)+","+str(f3)+","+str(f4)+","+str(f5)+","+str(l)+"\n")

if __name__ == "__main__":
    logisticRegression()