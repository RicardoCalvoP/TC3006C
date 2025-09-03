# Ricardo Calvo A01028889
# 09/2025
import os
import numpy
import seaborn
import math
import random
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
# -------------------- File functions
def read_file(file_path):
    X = []
    y = []
    data = []

    with open(file_path, "r", encoding="UTF-8") as file:
        for line in file:
            elements = line.strip().split(",")
            label = 0 if elements[-1] == "benign" else 1
            features = [float(v) for v in elements[:-1]]
            data.append((features, label))

    X = [features for features, _ in data]
    y = [label for _, label in data]

    return np.array(X), np.array(y)


def set_index(file, activation_funcs):
    file.write(
        "# Activity 2 logistic regression  with Scikit learn & manual\n\n")  # Title
    file.write("**Ricardo Calvo - A01028889**\n\n")  # Author
    file.write("## Table of Contents\n\n")  # Subtitle

    # Introduction
    file.write("1. [Introduction](#introduction)\n")  # Subtitle
    # Manual LR subtitles
    file.write("1. [Manual LR](#manual-lr)\n")  # Subtitle
    for activation_func in activation_funcs:
        file.write(
            f"   1. [Best {activation_func} results](#best-results-using-{activation_func.lower()}-activation-function)\n")

    # Sklearn LR subtitles
    # Subtitle
    file.write("2. [LR with Scikit learn](#lr-with-scikit-learn)\n")
    for activation_func in activation_funcs:
        file.write(
            f"   1. [Best {activation_func} results](#best-results-using-{activation_func.lower()}-activation-function-scikit-learn)\n")

    file.write("1. [Conclusion](#conclusion)\n")  # Subtitle


def write_introduction(file):
    file.write("## Introduction\n\n")
    file.write("In this report, we study the implementation and performance of the Logistic Regression "
               "algorithm using a manual implementation and the Scikit-learn library. "
               "Logistic Regression is a fundamental machine learning method widely applied to binary "
               "classification problems. It models the probability that a given input belongs to a "
               "specific class through the use of an activation function.\n\n")

    file.write("The dataset selected for this analysis is the Breast Cancer Wisconsin dataset, "
               "which contains clinical features that help distinguish between benign and malignant "
               "cases. For the manual implementation, the dataset is preprocessed by converting the "
               "class labels into numerical values (0 for benign and 1 for malignant), ensuring the "
               "correct input format for the algorithm.\n\n")

    file.write("The results will help compare the both approaches, "
               "as well as highlight the effect of activation functions and hyperparameters on the "
               "final model performance.\n\n")

    file.write("[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")

# -------------------- Manual LR functions
def sigmoid(z):
    return 1 / (1 + math.exp(-z))



def tanh(z):
    return math.tanh(z)

def gradient(sampleList, weights, activation_func):
    sumElements = 0.0

    for x, y in zip(sampleList, weights):
        sumElements += (x*y)

    return activation_func(sumElements)

def classifyList(testList, weights, activation_func):
    sumElements = 0
    # Multiply all features and optimized weights
    for x, y in zip(testList, weights):
        sumElements = sumElements+(x*y)
        # Obtain the sigmoid output which will tell us the class a test vector belongs
    out = activation_func(sumElements)
    p = out if activation_func is sigmoid else (out + 1)/2
    if p > 0.5:
        return 1.0
    else:
        return 0.

def stochasticGradientAscent(trainingLists, trainingLabels, featureNumber, activation_func, iterations=150, alpha_range=0.01):
    # Get the number of training samples
    sampleNumber = len(trainingLists)

    # Create a list of N features (featureNumber) for saving optimal weights (1.0 as initial value)
    weights = [1.0] * featureNumber
    # Iterate a fixed number of times for getting optimal weights
    for x in range(iterations):
        # Get the index number of training samples
        sampleIndex = list(range(sampleNumber))
        # For each training sample do the following
        for y in range(sampleNumber):
            """
            Alpha is the learning rate and controls how much the coefficients (and therefore the model)
            changes or learns each time it is updated.
            Alpha decreases as the number of iterations increases, but it never reaches 0
            """
            alpha = 4/(1.0+x+y)+alpha_range
            # Randomly obtain an index of one of training samples
            """
      Here, you’re randomly selecting each instance to use in updating the weights.
      This will reduce the small periodic variations that can be present if we analyze
      everything sequentially
      """
            randIndex = int(random.uniform(0, len(sampleIndex)))
            # Obtain the gradient from the current training sample and weights
            sampleGradient = gradient(trainingLists[randIndex], weights, activation_func)
            # Check the error rate
            error = trainingLabels[randIndex]-sampleGradient
            """
      we are calculating the error between the actual class and the predicted class and
      then moving in the direction of that error (CURRENT TRAINING PROCESS)
      """
            temp = []
            for index in range(featureNumber):
                temp.append(alpha*(error*trainingLists[randIndex][index]))

            for z in range(featureNumber):
                weights[z] = weights[z] + temp[z]

            del (sampleIndex[randIndex])
    return weights

# -------------------- Scikit learn LR functions

# -------------------- Helpers
def calc_acc(X, y, optimalWeights, activation_func):
    correctPredictions = 0
    for x, y_hat in zip(X, y):
        predicted = classifyList(x, optimalWeights, activation_func)
        if predicted == y_hat:
            correctPredictions = correctPredictions + 1

    return correctPredictions / len(y) * 100


# -------------------- Main function
def logistic_regression():
    # Load training and test data
    training_X, training_y = read_file(
        "Homeworks/DataScience/LR/cancerTraining.txt")
    # Load test data
    test_X, test_y = read_file("Homeworks/DataScience/LR/cancerTest.txt")
    # Number of repetitions for optimizing the weights
    min_iterations = 1
    max_iterations = 100
    # Ranges of alpha
    min_range_alpha = 0.01
    max_range_alpha = 0.3
    # Number of features found in the dataset
    featureNumber = len(training_X[0])

    manual_activation_funcs = {
    "Sigmoid": sigmoid,
    "Tanh": tanh
    }

    # -------------------- Manual testing
    current_manual_weights = []
    for name, func in manual_activation_funcs.items():
        print(f"Using {name}")
        current_manual_weights = stochasticGradientAscent(
        training_X, training_y, featureNumber, func, max_iterations, max_range_alpha)
        accuracy = calc_acc(test_X, test_y, current_manual_weights, func)
        print("Model accuracy: "+str(accuracy)+"%")

    # -------------------- Scikit learn testing
    for iteration in range(min_iterations, max_iterations):
        # print("Iteration: " + str(iteration))
        curr_range_alpha = min_range_alpha
        while curr_range_alpha <= max_range_alpha:
            # print("Range alhpa: " + str(curr_range_alpha))
            pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                random_state=0, max_iter=iteration, C=curr_range_alpha,solver="lbfgs"  # binaria
    ))
    ])
            pipe.fit(training_X, training_y)
            y_pred = pipe.predict(test_X)
            acc = accuracy_score(test_y, y_pred) * 100
            # print(f"[sklearn] Accuracy: {acc:.2f}%")

            curr_range_alpha = curr_range_alpha + 0.01

    # alpha_range = alpha_range + 0.01


    # -------------------- Create report
    filepath = "Homeworks/DataScience/LR/logistic_regression_a01028889.md"
    # If file exists, remove it
    if os.path.exists(filepath):
        os.remove(filepath)

    # Write results to the file
    with open(filepath, "w", encoding="UTF-8") as file:
        set_index(file, list(manual_activation_funcs.keys()))
        write_introduction(file)


if __name__ == "__main__":
    logistic_regression()
