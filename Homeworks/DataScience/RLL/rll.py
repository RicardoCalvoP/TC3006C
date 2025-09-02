# Ricardo Calvo A01028889
# 09/2025
import os
import numpy
import seaborn
import math
import random
import pandas as pd
import sklearn
import matplotlib as plt


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

    random.shuffle(data)

    X = [features for features, _ in data]
    y = [label for _, label in data]

    return X, y


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

# Main function to execute RL


def logistic_regression():
    # Load training and test data
    traing_X, traing_y = read_file(
        "Homeworks/DataScience/RLL/cancerTraining.txt")
    # Load test data
    test_X, test_y = read_file("Homeworks/DataScience/RLL/cancerTest.txt")
    # Initialize helper variables
    activation_funcs = ["sigmoide", "tanh"]
    # Create result file
    filepath = "Homeworks/DataScience/RLL/logistic_regression_a01028889.md"

    # If file exists, remove it
    if os.path.exists(filepath):
        os.remove(filepath)

    # Write results to the file
    with open(filepath, "w", encoding="UTF-8") as file:
        set_index(file, activation_funcs)
        write_introduction(file)


if __name__ == "__main__":
    logistic_regression()
