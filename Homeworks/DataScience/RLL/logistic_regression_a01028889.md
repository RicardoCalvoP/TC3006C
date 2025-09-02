# Activity 2 logistic regression  with Scikit learn & manual

**Ricardo Calvo - A01028889**

## Table of Contents

1. [Introduction](#introduction)
1. [Manual LR](#manual-lr)
   1. [Best sigmoide results](#best-results-using-sigmoide-distance-metric)
   1. [Best tanh results](#best-results-using-tanh-distance-metric)
2. [LR with Scikit learn](#lr-with-scikit-learn)
   1. [Best sigmoide results](#best-results-using-sigmoide-distance-metric-scikit-learn)
   1. [Best tanh results](#best-results-using-tanh-distance-metric-scikit-learn)
1. [Conclusion](#conclusion)
## Introduction

In this report, we study the implementation and performance of the Logistic Regression algorithm using a manual implementation and the Scikit-learn library. Logistic Regression is a fundamental machine learning method widely applied to binary classification problems. It models the probability that a given input belongs to a specific class through the use of an activation function.

The dataset selected for this analysis is the Breast Cancer Wisconsin dataset, which contains clinical features that help distinguish between benign and malignant cases. For the manual implementation, the dataset is preprocessed by converting the class labels into numerical values (0 for benign and 1 for malignant), ensuring the correct input format for the algorithm.

The results will help compare the both approaches, as well as highlight the effect of activation functions and hyperparameters on the final model performance.

[Return to Table of Contents](#table-of-contents)

 --- 

