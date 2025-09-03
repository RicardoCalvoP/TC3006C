# Ricardo Calvo A01028889
# 09/2025
import os
import numpy
import seaborn
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

import warnings
from sklearn.exceptions import ConvergenceWarning
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
    # Sklearn LR subtitles
    # Subtitle
    file.write("2. [Scikit learn LR](#scikit-learn-lr)\n")

    file.write("1. [Conclusion](#conclusion)\n")  # Subtitle


def write_introduction(file):
    file.write("## Introduction\n\n")
    file.write(
        "In this report, we study the implementation and performance of the Logistic Regression "
        "algorithm using a manual implementation and the Scikit-learn library. "
        "Logistic Regression is a fundamental machine learning method widely applied to binary "
        "classification problems. It models the probability that a given input belongs to a "
        "specific class through the use of an activation function.\n\n")

    file.write(
        "The dataset selected for this analysis is the Breast Cancer Wisconsin dataset, "
        "which contains clinical features that help distinguish between benign and malignant "
        "cases. For the manual implementation, the dataset is preprocessed by converting the "
        "class labels into numerical values (0 for benign and 1 for malignant), ensuring the "
        "correct input format for the algorithm.\n\n")

    file.write(
        "The results will help compare the both approaches, "
        "as well as highlight the effect of activation functions and hyperparameters on the "
        "final model performance.\n\n")

    file.write(
        "For the manual approach, we use part of the code provided by the professor "
        "with two activation functions: the standard sigmoid and, as our proposal, the tanh function.\n\n"
        "In this approach, we will set the number of iterations from 1 to 100, and for each iteration count "
        "we also set the learning rate parameter (alpha) in a range from 0.01 to 0.3.\n\n"
        "We will do the same experiment using the scikit-learn implementation. "
        "Since scikit-learn does not allow direct control over the learning rate (alpha), "
        "we vary the regularization parameter C instead, while also testing iteration counts from 1 to 100.\n\n"
    )

    file.write("[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")


def write_results(file, impl_name, best_acc, best_prec, best_rec, avg_acc, avg_prec, avg_rec, graph_func, cm_avg, avg_acc_tanh="", avg_prec_tanh="", avg_rec_tanh="", cm_avg_tanh=""):
    file.write(f"## {impl_name} LR\n\n")

    file.write(
        f"In the case of the **{impl_name} implementation**, "
        "we evaluated different configurations by varying the number of iterations and the learning parameter "
        "(alpha for the manual approach or the regularization parameter C for the scikit-learn implementation). "
        "Two activation functions were tested: the standard sigmoid function and the tanh function.\n\n"
    )

    # Best results
    file.write("### Best Results\n")
    file.write(
        f"The best overall **accuracy** was {best_acc[0]*100:.2f}%, "
        f"achieved using the **{best_acc[1]}** activation function, "
        f"with {best_acc[2]} iterations and an alpha/C value of {best_acc[3]:.2f}.\n\n"
    )
    file.write(
        f"For **precision**, the highest value obtained was {best_prec[0]*100:.2f}%, "
        f"with the {best_prec[1]} activation function, "
        f"{best_prec[2]} iterations, and alpha/C = {best_prec[3]:.2f}. "
        "A precision this high with such a small number of iterations in our experiments "
        "suggests that the model was very effective at avoiding false positives under that configuration. "
        "Nevertheless, results of this kind often arise when the model predicts only a limited number of positive cases, "
        "so the outcome should be interpreted with caution in terms of generalization.\n\n"
    )

    file.write(
        f"The best **recall** reached was {best_rec[0]*100:.2f}%, "
        f"using the {best_rec[1]} activation function, "
        f"with {best_rec[2]} iterations when alpha/C = {best_rec[3]:.2f}. "
        "A recall this high in such an early stage of training indicates that the model was able "
        "to correctly capture nearly all of the actual positive cases. "
        "However, in practice this can also happen when the model tends to classify most inputs as positives, "
        "which increases sensitivity but may come at the cost of precision. "
        "This highlights the importance of considering multiple metrics together "
        "to assess the overall quality of the model.\n\n"
    )

    # Average results
    file.write("### Average Results\n")
    file.write(
        "When averaging the performance over the full range of iterations and alpha/C values, "
        f"the model obtained an **average accuracy of {avg_acc * 100:.2f}%**, "
        f"an **average precision of {avg_prec*100:.2f}%**, and "
        f"an **average recall of {avg_rec*100:.2f}%**. "
        "These results suggest that while the model can reach strong performance under optimal settings, "
        "its overall stability across all configurations is slightly lower.\n\n"
    )

    # Confusion matrix
    file.write("### Average Confusion Matrix\n")
    file.write(
        "The following confusion matrix shows the average counts of true positives, false positives, "
        "true negatives, and false negatives across all runs. "
        "This provides a global view of the classification performance of the model:\n\n"
    )
    name = "Manual Sigmoid" if impl_name == "Manual" else impl_name
    graph_func(file, cm_avg, f"{name} Average Confusion Graph")

    if impl_name == "Manual":
        file.write("### Average Results (Tanh)\n")
        file.write(
            f"Using the **tanh** activation function, the model reached an "
            f"**average accuracy of {avg_acc_tanh*100:.2f}%**, "
            f"an **average precision of {avg_prec_tanh*100:.2f}%**, and "
            f"an **average recall of {avg_rec_tanh*100:.2f}%**. "
            "Compared with the sigmoid results, these values highlight how the choice of activation "
            "function can slightly alter the trade-off between precision and recall, "
            "even when the same range of iterations and alpha/C values is used.\n\n"
        )

        file.write("### Average Confusion Matrix (Tanh)\n")
        file.write(
            "The confusion matrix below summarizes the averaged classification outcomes when using tanh. "
            "By contrasting it with the sigmoid-based matrix, one can observe whether tanh tends to favor "
            "recall (capturing more true positives) or precision (avoiding false positives) under similar conditions:\n\n"
        )
        graph_func(file, cm_avg_tanh,
                   f"{impl_name} Tanh Average Confusion Graph")

    file.write("[Return to Table of Contents](#table-of-contents)\n\n --- \n\n")

    # -------------------- Manual LR functions


def sigmoid(z):
    if z >= 0:
        ez = np.exp(-z)
        return 1/(1+ez)
    else:
        ez = np.exp(z)
        return ez/(1+ez)


def tanh(z):
    return np.tanh(z)


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
            idx = sampleIndex[randIndex]

            out = gradient(trainingLists[idx], weights, activation_func)
            p = out if activation_func is sigmoid else (
                out + 1)/2  # mapear tanh→[0,1]
            error = trainingLabels[idx] - p

            temp = []
            for j in range(featureNumber):
                temp.append(alpha * error * trainingLists[idx][j])

            for j in range(featureNumber):
                weights[j] += temp[j]

            del sampleIndex[randIndex]

    return weights

# -------------------- Scikit learn LR functions

# -------------------- Helpers


def predict(X, optimalWeights, activation_func):
    y_predicted = []
    for x in X:
        predicted = classifyList(x, optimalWeights, activation_func)
        y_predicted.append(predicted)

    return np.array(y_predicted)


def get_best_values(cur_value, best_values, act_func, iteration, alpha_range):
    if cur_value > best_values[0]:
        best_values[0] = cur_value
        best_values[1] = act_func
        best_values[2] = iteration
        best_values[3] = alpha_range

    return best_values

# -------------------- Graphs functions


def plot_confusion_matrix(file, cm_avg, title):

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_avg, display_labels=["0", "1"])
    disp.plot(cmap="Blues", values_format=".2f", colorbar=True)
    plt.title(title)
    plt.tight_layout()

    path = "Homeworks/DataScience/LR/Graphs/"
    os.makedirs(path, exist_ok=True)
    filename = f"{title.lower().replace(' ', '_')}_confusion_matrix.png"
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    file.write(f"![Confusion Matrix](Graphs/{filename})\n\n")
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

    total_runs = 0

    # Number of features found in the dataset
    featureNumber = len(training_X[0])

    # -------------------- Manual variabels
    manual_activation_funcs = {
        "Sigmoid": sigmoid,
        "Tanh": tanh
    }

    manual_sig_acc_sum = manual_sig_prec_sum = manual_sig_rec_sum = 0.0
    manual_sig_cms = []

    manual_tanh_acc_sum = manual_tanh_prec_sum = manual_tanh_rec_sum = 0.0
    manual_tanh_cms = []

    # [value, optimal act func, optimal iterations, optimal alpha range]
    manual_acc_best_data = [0, '', 0, 0]
    # [value, optimal act func, optimal iterations, optimal alpha range]
    manual_pres_best_data = [0, '', 0, 0]
    # [value, optimal act func, optimal iterations, optimal alpha range]
    manual_recall_best_data = [0, '', 0, 0]

    # -------------------- Scikit learn variables

    skl_acc_sum, skl_prec_sum, skl_rec_sum = 0, 0, 0
    skl_cms = []
    # [value, optimal act func, optimal iterations, optimal alpha range]
    skl_acc_best_data = [0, '', 0, 0]
    # [value, optimal act func, optimal iterations, optimal alpha range]
    skl_pres_best_data = [0, '', 0, 0]
    # [value, optimal act func, optimal iterations, optimal alpha range]
    skl_recall_best_data = [0, '', 0, 0]

    # -------------------- Changing from 1 - 100 iterations & from range of alpha from 0.01 to 0.3
    for iteration in range(min_iterations, max_iterations + 1):
        print("Current Iteration: " + str(iteration))
        curr_range_alpha = min_range_alpha
        while curr_range_alpha <= max_range_alpha:

            # -------------------- Manual testing
            current_manual_weights = []
            for name, func in manual_activation_funcs.items():
                # Get optimal weights
                current_manual_weights = stochasticGradientAscent(
                    training_X, training_y, featureNumber, func, iteration, curr_range_alpha)
                # With the weigths get predicted output
                y_pred = predict(test_X, current_manual_weights, func)
                # Calculate accuracy of model
                acc = accuracy_score(test_y, y_pred)
                # Calculate precision of model
                prec = precision_score(test_y, y_pred)
                # Calculate recall of model
                rec = recall_score(test_y, y_pred)
                # Calculate confusion matrix
                cm = confusion_matrix(test_y, y_pred)

                if name == "Sigmoid":
                    manual_sig_acc_sum += acc
                    manual_sig_prec_sum += prec
                    manual_sig_rec_sum += rec
                    manual_sig_cms.append(cm)
                else:  # Tanh
                    manual_tanh_acc_sum += acc
                    manual_tanh_prec_sum += prec
                    manual_tanh_rec_sum += rec
                    manual_tanh_cms.append(cm)
                # Change best scores if needed
                # Compare best accuracy with current accuracy
                manual_acc_best_data = get_best_values(
                    acc, manual_acc_best_data, name, iteration, curr_range_alpha)
                # Compare best precision with current precision
                manual_pres_best_data = get_best_values(
                    prec, manual_pres_best_data, name, iteration, curr_range_alpha)
                # Compare best recall with current recall
                manual_recall_best_data = get_best_values(
                    rec, manual_recall_best_data, name, iteration, curr_range_alpha)

            # -------------------- Scikit learn testing
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    random_state=0, max_iter=iteration, C=curr_range_alpha, solver="lbfgs"  # binaria
                ))
            ])
            # Train the model
            pipe.fit(training_X, training_y)
            # Predict the output
            y_pred = pipe.predict(test_X)
            # Calculate accuracy of model
            acc = accuracy_score(test_y, y_pred)
            # Calculate precision of model
            prec = precision_score(test_y, y_pred)
            # Calculate recall of model
            rec = recall_score(test_y, y_pred)
            # Calculate confusion matrix
            cm = confusion_matrix(test_y, y_pred)
            skl_cms.append(cm)
            # Sum preducts for the avgs
            skl_acc_sum = skl_acc_sum + acc
            skl_prec_sum = skl_prec_sum + prec
            skl_rec_sum = skl_rec_sum + rec
            # Compare best accuracy with current accuracy
            skl_acc_best_data = get_best_values(
                acc, skl_acc_best_data, '', iteration, curr_range_alpha)
            # Compare best precision with current precision
            skl_pres_best_data = get_best_values(
                prec, skl_pres_best_data, '', iteration, curr_range_alpha)
            # Compare best recall with current recall
            skl_recall_best_data = get_best_values(
                rec, skl_recall_best_data, '', iteration, curr_range_alpha)

            # Sum total runs
            total_runs = total_runs + 1
            # Skip to next range of alpha
            curr_range_alpha = curr_range_alpha + 0.01

    # -------------------- Get averages
    manual_sig_avg_acc = manual_sig_acc_sum / total_runs
    manual_sig_avg_prec = manual_sig_prec_sum / total_runs
    manual_sig_avg_rec = manual_sig_rec_sum / total_runs
    manual_sig_avg_cm = np.mean(np.stack(manual_sig_cms, axis=0), axis=0)

    manual_tanh_avg_acc = manual_tanh_acc_sum / total_runs
    manual_tanh_avg_prec = manual_tanh_prec_sum / total_runs
    manual_tanh_avg_rec = manual_tanh_rec_sum / total_runs
    manual_tanh_avg_cm = np.mean(np.stack(manual_tanh_cms, axis=0), axis=0)

    skl_avg_acc = skl_acc_sum / total_runs
    skl_avg_prec = skl_prec_sum / total_runs
    skl_avg_rec = skl_rec_sum / total_runs
    skl_avg_cm = np.mean(np.stack(skl_cms, axis=0), axis=0)

    # -------------------- Create report
    filepath = "Homeworks/DataScience/LR/logistic_regression_a01028889.md"
    # If file exists, remove it
    if os.path.exists(filepath):
        os.remove(filepath)

    # Write results to the file
    with open(filepath, "w", encoding="UTF-8") as file:
        set_index(file, list(manual_activation_funcs.keys()))
        write_introduction(file)
        # Write manual resutls
        write_results(file, "Manual", manual_acc_best_data,
                      manual_pres_best_data, manual_recall_best_data,
                      manual_sig_avg_acc, manual_sig_avg_prec, manual_sig_avg_rec,
                      plot_confusion_matrix, manual_sig_avg_cm, manual_tanh_avg_acc,
                      manual_tanh_avg_prec, manual_tanh_avg_rec, manual_tanh_avg_cm)
        # Write skl resutls
        write_results(file, "Scikit learn", skl_acc_best_data,
                      skl_pres_best_data, skl_recall_best_data,
                      skl_avg_acc, skl_avg_prec, skl_avg_rec,
                      plot_confusion_matrix, skl_avg_cm)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    logistic_regression()
