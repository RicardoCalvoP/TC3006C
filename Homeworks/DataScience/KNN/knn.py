# Ricardo Calvo A01028889
# 08/2025
import os
import numpy as np
import seaborn as sns
import pandas as pd
import math
from statistics import mode
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# Reads the data from a file and returns two arrays:
def read_file(file_path):
    X = []
    y = []
    with open(file_path, "r", encoding="UTF-8") as file:
        for line in file:
            elements = line.strip().split(",")
            features = [float(v) for v in elements[:-1]]
            temp = elements[-1]
            if temp == "Absence":
                label = 0
            else:
                label = 1
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Sets Index for file titles and subtitles
def set_index(file, distance_metrics):
    file.write("# Activity 1 KNN with Scikit learn & manual\n\n") # Title
    file.write("**Ricardo Calvo - A01028889**\n\n") # Author
    file.write("## Table of Contents\n\n") # Subtitle

    # Introduction
    file.write("1. [Introduction](#introduction)\n") # Subtitle
    # Manual KNN subtitles
    file.write("1. [Manual KNN](#manual-knn)\n") # Subtitle
    for metric in distance_metrics:
        file.write(f"   1. [Best {metric} results](#best-results-using-{metric.lower()}-distance-metric)\n")

    # Sklearn KNN subtitles
    file.write("2. [KNN with Scikit learn](#knn-with-scikit-learn)\n") # Subtitle
    for metric in distance_metrics:
        file.write(f"   1. [Best {metric} results](#best-results-using-{metric.lower()}-distance-metric-scikit-learn)\n")

    file.write("1. [Conclusion](#conclusion)\n") # Subtitle


def write_introduction(file):
    file.write("## Introduction\n\n") # Subtitle
    file.write("In this report, we explore the implementation and performance of the K-Nearest Neighbors"
                "(KNN) algorithm using both a manual approach and the Scikit-learn library. "
                "The KNN algorithm is a simple machine learning technique used for "
                "classification tasks. It operates on the principle that similar data points are likely "
                "to belong to the same class.\n\n")
    file.write("The dataset used in this analysis contains various features related to heart health, "
               "with the goal of predicting the presence or absence of heart disease. "
               "First we are going to test the accuracy of the models using all features, getting the best" \
               "model based on the accuracy, our program will return the best distance metric (euclidean, manhattan or " \
               "minkowski) and the best k value (from 3 to 41). Then, we will evaluate the contribution of each feature " \
               "by systematically removing one feature at a time and observing the impact on model performance. "
               )

    # return to table of contents
    file.write("\n\n[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")

def write_results(file ,current_distance, best_k, best_accuracy, y_pred, y_real, implementation = ""):
    file.write(f"## Best results using {current_distance} distance metric {implementation}\n\n")
    file.write(f"### k={best_k}\n\n")

    # Table results header
    file.write("| Predicted | Real Value | Correct? |\n")
    file.write("|-----------|------------|----------|\n")
    for predicted, y in zip(y_pred, y_real):
        file.write(f"| {predicted} | {y} | {predicted == y } |\n")
    file.write(f"\n**Model accuracy: {best_accuracy:.4f}%** \n\n ")
    file.write("[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")

def write_conclusion(file, best_distance, best_k, best_accuracy, model, filtered_manual_results, filtered_manual_features, filtered_sclrn_results, filtered_sclrn_features):
    file.write("## Conclusion\n\n")
    file.write("After completing all test we got that the best way to evaluate" \
    f"the probability of someone having heart issues is by using **{model}** since " \
    f"is the one with the bigger accuracy on **{best_accuracy:.4f}%**, when **k = {best_k}**" \
    f" calculating the distance with **{best_distance}** metric distance.\n\n"\
    "As we can see in the Scikit-learn performance graph, there was no significant difference "
    "in accuracy when using the Euclidean, Manhattan, or Minkowski distance metrics. " \
    "This suggests that for this specific dataset, the  feature " \
    "distribution is such that the method of calculating distance between data has a " \
    "minimal effect on the KNN model's predictive capability. \n\n" \
    f"However we were able to get our accuracy to **{filtered_manual_results[2]:.4f}%** with our manual "
    f"functions using **{filtered_manual_results[0]}** metric distance " \
    f"when **k = {filtered_manual_results[1]}** using the folowwing data: ")

    for feature in filtered_manual_features:
        file.write(f"{feature}, ")

    file.write("on the other hand, by using Scikit-learn functions we could get our " \
    f"accuracy to **{filtered_sclrn_results[2]:.4f}%** when using " \
    f"{filtered_sclrn_results[0]} metric distance, when **k = " \
    f"{filtered_sclrn_results[1]}** when we use the following data: ")

    for feature in filtered_sclrn_features:
        file.write(f"{feature}, ")

    file.write("\n\n[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")



def create_accuracy_graph(file, accuracies, current_distance, implementation = ""):
    # Graph to check Accuracy based on K
    # create a array from 3 - 41
    ks = list(range(3, 42))
    plt.figure(figsize=(8,5))
    plt.plot(ks, accuracies, marker="o")
    plt.title(f"{current_distance} accuracy vs K")
    plt.xlabel("K")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    path = "Homeworks/DataScience/KNN/Graphs/"
    os.makedirs(path, exist_ok=True)   # asegúrate que la carpeta exista
    filename = f"{current_distance}_accuracy_plot{implementation}.png"
    filepath = os.path.join(path, filename)

    # Si ya existe el archivo, lo elimina
    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    file.write(f"![Accuracy vs K](Graphs/{filename})\n\n")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def create_confusion_matrix(file, y_true, y_pred, current_distance, implementation=""):
    plt.figure(figsize=(8,5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({current_distance} {implementation})")

    path = "Homeworks/DataScience/KNN/Graphs/"
    os.makedirs(path, exist_ok=True)
    filename = f"{current_distance}_confusion_matrix{implementation}.png"
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    file.write(f"![Confusion Matrix](Graphs/{filename})\n\n")

def create_heatmap(file, X, headers):

    file.write("## Heatmap")
    file.write("\n\nWe can see with the following graph the " \
    "behavior between the data within our dataframe")

    df = pd.DataFrame(X, columns=headers)

    corr = df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")

    path = "Homeworks/DataScience/KNN/Graphs/"
    os.makedirs(path, exist_ok=True)
    filename = "correlation_heatmap.png"
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    # Insertar imagen al reporte
    file.write(f"![Correlation Heatmap](Graphs/{filename})\n\n")
    file.write(
    "\n\nBased on our heatmap, darker blue values represent strong negative correlations "
    "(close to -1), meaning that when one feature increases, the other tends to decrease. "
    "On the other hand, darker red values represent strong positive correlations "
    "(close to +1), where both features increase or decrease together. "
    "Values close to white (around 0) indicate little or no linear correlation.\n\n"
)

    file.write("[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")



def create_performance_graph(file, accuracies, distance_metric):
    plt.figure(figsize=(8,5))
    plt.bar(distance_metric, accuracies,)

    # Añadir etiquetas y título
    plt.xlabel('Distance Metrics')
    plt.ylabel('Accuracy')
    plt.title("Scikit-learn performance")

    path = "Homeworks/DataScience/KNN/Graphs/"
    os.makedirs(path, exist_ok=True)
    filename = "scikit_learn_performance.png"
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    file.write(f"![Scikit learn performance](Graphs/{filename})\n\n")

def filter_features_by_contribution(train_X, train_y, test_X, test_y, distance_metric, headers, k):
    # Variables to track the best results
    manual_best_results = [0, 0, 0]  # [best_metric, best_k, best_accuracy]
    manual_best_features = []
    sklearn_best_results = [0, 0, 0]  # [best_metric, best_k, best_accuracy]
    sklearn_best_features = []

    # Loop through each feature to be removed (represented by index 'i')
    for i in range(train_X.shape[1]):
        # Create a new dataset with a single feature removed
        filtered_train_X = np.delete(train_X, i, axis=1)
        filtered_test_X = np.delete(test_X, i, axis=1)

        # Get the list of features that remain
        current_features = headers[:i] + headers[i+1:]

        # Call the existing KNN functions with the filtered data.
        # These functions will internally find the best k (3 to 41) for this specific filtered dataset.
        temp_mbr = manual_knn(filtered_train_X, train_y, filtered_test_X, test_y, k, distance_metric, None)
        temp_skbr = sklearn_knn(filtered_train_X, train_y, filtered_test_X, test_y, k, distance_metric, None)

        # Compare the best result from this run with the overall best found so far
        if temp_mbr[2] > manual_best_results[2]:
            manual_best_results = temp_mbr
            manual_best_features = current_features

        if temp_skbr[2] > sklearn_best_results[2]:
            sklearn_best_results = temp_skbr
            sklearn_best_features = current_features

    return manual_best_results, manual_best_features, sklearn_best_results, sklearn_best_features

def euclideanDistance(list1,list2):
    sum_list = 0
    for x,y in zip(list1,list2):
        sum_list += ((y-x)**2)
    return math.sqrt(sum_list)

def manhattanDistance(list1,list2):
    sum_list = 0
    for x,y in zip(list1,list2):
        sum_list += abs(y-x)
    return sum_list

def minkowskiDistance(list1,list2,p):
    sum_list = 0
    for x,y in zip(list1,list2):
        sum_list += abs(y-x)**p
    return sum_list**(1/p)

def classify(test_list, training_lists,training_labels, k, distance_metric):
    distance = []
    for training_list, label in zip (training_lists, training_labels):
        value = 0
        if distance_metric == "Euclidean":
            value = euclideanDistance(test_list,training_list)
        elif distance_metric == "Manhattan":
            value = manhattanDistance(test_list,training_list)
        else:
            value = minkowskiDistance(test_list,training_list,3)
        distance.append((value, label))

    distance.sort(key=itemgetter(0))
    vote_labels = []

    for x in distance[:k]:
        vote_labels.append(x[1])

    return mode(vote_labels)

def manual_knn(traing_X, traing_y, test_X, test_y, k, distance_metric, file):
    # Variable for best accuracy results
    global_best_model = ''
    global_best_k = 0
    global_best_accuracy = 0
    current_accuracy = 0

    totalPredictions = len(test_y)
    if file:
        file.write("## Manual KNN\n\n") # Subtitle

    for metric in distance_metric:
        metric_best_k = 0
        metric_best_accuracy = 0
        metric_best_predictions = []
        accuracies = []
        temp_k = k
        # Repeat for every k from 3 to 41
        while temp_k <=41:
            y_pred = []
            correctPredictions = 0
            # Loop through test data
            for x,y in zip (test_X, test_y):
                # Get prediction based on k and distance metric
                predicted = classify(x, traing_X, traing_y, temp_k, metric)
                y_pred.append(predicted)
                # Count correct predictions for accuracy calculation
                if predicted==y:
                    correctPredictions+=1

            # After all predictions, write accuracy to the file
            current_accuracy = (correctPredictions/totalPredictions)*100
            accuracies.append(current_accuracy)

            #Check for metric best
            if current_accuracy > metric_best_accuracy:
                metric_best_accuracy = current_accuracy
                metric_best_k = temp_k
                metric_best_predictions = y_pred
            # Check for best model
            if current_accuracy > global_best_accuracy:
                global_best_accuracy = current_accuracy
                global_best_model = metric
                global_best_k = temp_k


            # Set counters for next k
            temp_k += 1
        if file:
            write_results(file, metric, metric_best_k, metric_best_accuracy, metric_best_predictions, test_y)
            create_accuracy_graph(file, accuracies, metric, "manual")
            create_confusion_matrix(file, test_y, metric_best_predictions, metric, "manual")

    return [global_best_model, global_best_k, global_best_accuracy]

def sklearn_knn(traing_X, traing_y, test_X, test_y, k, distance_metric, file):
    # Variable for best accuracy results
    global_best_model = ''
    global_best_k = 0
    global_best_accuracy = 0
    current_accuracy = 0
    best_accuracies_based_on_metrics = []

    totalPredictions = len(test_y)

    if file:
        file.write("## KNN with Scikit learn\n\n") # Subtitle

    for metric in distance_metric:
        metric_best_k = 0
        metric_best_accuracy = 0
        metric_best_predictions = []
        accuracies = []
        temp_k = k
        # Repeat for every k from 3 to 41
        while temp_k <=41:
            correctPredictions = 0

            model = KNeighborsClassifier(n_neighbors=temp_k, metric=metric.lower())
            model.fit(traing_X, traing_y)
            predictions = model.predict(test_X)

            for x, y in zip(predictions, test_y):
                if x == y:
                    correctPredictions = correctPredictions + 1
            # After all predictions, write accuracy to the file
            current_accuracy = (correctPredictions/totalPredictions)*100
            accuracies.append(current_accuracy)

            #Check for metric best
            if current_accuracy > metric_best_accuracy:
                metric_best_accuracy = current_accuracy
                metric_best_k = temp_k
                metric_best_predictions = predictions
            # Check for best model
            if current_accuracy > global_best_accuracy:
                global_best_accuracy = current_accuracy
                global_best_model = metric
                global_best_k = temp_k

            temp_k += 1
        best_accuracies_based_on_metrics.append(metric_best_accuracy)

        if file:
            write_results(file, metric, metric_best_k, metric_best_accuracy, metric_best_predictions, test_y, "scikit learn")
            create_accuracy_graph(file, accuracies, metric, "scikit_learn")
            create_confusion_matrix(file, test_y, metric_best_predictions, metric, "scikit_learn")

    if file:
        create_performance_graph(file, best_accuracies_based_on_metrics, distance_metric)

    return [global_best_model, global_best_k, global_best_accuracy]

# Main function to execute KNN
def knn():
    # Load training and test data
    traing_X, traing_y = read_file("Homeworks/DataScience/KNN/training.txt")
    # Load test data
    test_X, test_y = read_file("Homeworks/DataScience/KNN/test.txt")

    # Initialize main variables
    k = 3 # number of neighbors

    # Initialize helper variables
    distance_metrics = ["Euclidean", "Manhattan", "Minkowski"]

    headers = ["Age", "Gender", "Chest pain", "Arterial pressure",
               "Cholesterol", "Glucose", "EGG", "Heart rate", "Exercise angina",
               "ST depression","ST Segment", "Number of vessels", "Thalassemia"]

    # Create result file
    filepath = "Homeworks/DataScience/KNN/knn_a01028889.md"

    # If file exists, remove it
    if os.path.exists(filepath):
        os.remove(filepath)

    # Write results to the file
    with open(filepath, "w", encoding="UTF-8") as file:
        set_index(file, distance_metrics)
        write_introduction(file)
        create_heatmap(file, traing_X, headers)
        manual_results = manual_knn(traing_X, traing_y, test_X, test_y, k, distance_metrics, file)
        sklearn_reasults = sklearn_knn(traing_X, traing_y, test_X, test_y, k, distance_metrics, file)
        filtered_manual_best_results, filtered_manual_features, filtered_sclrn_results, filtered_sclrn_features = filter_features_by_contribution(traing_X, traing_y, test_X, test_y, distance_metrics, headers, k)
        if manual_results[2] <= sklearn_reasults[2]:
            write_conclusion(file, sklearn_reasults[0], sklearn_reasults[1], sklearn_reasults[2], "sckit-learn", filtered_manual_best_results, filtered_manual_features, filtered_sclrn_results, filtered_sclrn_features)
        else:
            write_conclusion(file, manual_results[0], manual_results[1], manual_results[2], "manual", filtered_manual_best_results, filtered_manual_features, filtered_sclrn_results, filtered_sclrn_features)

        file.close()



if __name__ == "__main__":
    knn()