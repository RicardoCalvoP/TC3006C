# 09/2025
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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
        "We selected the tanh function as an alternative because, unlike the sigmoid, it is zero-centered, "
        "producing outputs in the range of -1 to 1. This property can help the optimization process converge faster "
        "in some cases and reduce issues with gradients being biased toward positive values. "
        "By comparing both activation functions, we aimed to evaluate whether this theoretical advantage of tanh "
        "translates into better performance in practice.\n\n"
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

def write_conclusion(file):
    file.write("## Conclusion\n\n")
    file.write(
        "Throughout this work we implemented logistic regression in two ways: a manual version using sigmoid and tanh "
        "activation functions, and a scikit-learn implementation using the standard library solver. "
        "Both approaches allowed us to explore the impact of iteration counts, learning parameters, and activation choices "
        "on the performance of the model when applied to breast cancer classification.\n\n"
    )

    file.write(
        "The manual implementation proved useful as an educational tool, showing how changes in the learning rate or the number "
        "of iterations affect the stability of accuracy, precision, and recall. The inclusion of tanh provided an additional perspective, "
        "demonstrating how a zero-centered activation function can influence the optimization process. However, the manual results also "
        "revealed some limitations, including more variability in performance and the presence of configurations that could easily overfit "
        "or underperform depending on parameter choices.\n\n"
    )

    file.write(
        "On the other hand, the scikit-learn implementation consistently achieved stronger and more stable results. "
        "Its ROC curve showed higher sensitivity with fewer false positives, and the average confusion matrix confirmed a very low "
        "rate of false negatives — a crucial aspect in the context of breast cancer detection. The probability distributions were better "
        "separated between classes, and the feature importance plot highlighted consistent predictors without contradictory contributions. "
        "Altogether, these results suggest that while the manual approach helps us understand the inner mechanics of logistic regression, "
        "the scikit-learn implementation provides more reliable and clinically applicable outcomes.\n\n"
    )

    file.write(
        "In conclusion, scikit-learn not only simplifies the training process but also enhances performance stability, "
        "making it the preferred option when the objective is to deploy logistic regression in real-world medical problems "
        "where minimizing false negatives is essential. The manual model, however, remains a valuable didactic resource "
        "for understanding the algorithm’s behavior and the influence of its hyperparameters.\n\n"
    )


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

def predict_proba_manual(X, optimalWeights, activation_func):
    y_probs = []
    for x in X:
        z = sum(w * xi for w, xi in zip(optimalWeights, x))
        p = activation_func(z) if activation_func is sigmoid else (activation_func(z) + 1) / 2
        y_probs.append(p)
    return np.array(y_probs)


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

# -------------------- Helpers

def predict(X, optimalWeights, activation_func):
    y_predicted = []
    for x in X:
        predicted = classifyList(x, optimalWeights, activation_func)
        y_predicted.append(int(predicted))

    return np.array(y_predicted)

def get_best_values(cur_value, best_values, act_func, iteration, alpha_range):
    if cur_value > best_values[0]:
        best_values[0] = cur_value
        best_values[1] = act_func
        best_values[2] = iteration
        best_values[3] = alpha_range
        return best_values, True

    return best_values, False

# -------------------- Graphs functions
def plot_confusion_matrix(file, cm_avg, title):

    plt.figure(figsize=(6, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_avg, display_labels=["0", "1"])
    disp.plot(cmap="Blues", values_format=".2f", colorbar=True)
    plt.title(title)
    plt.tight_layout()

    path = "Graphs/"
    os.makedirs(path, exist_ok=True)
    filename = f"{title.lower().replace(' ', '_')}_confusion_matrix.png"
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    file.write(f"![Confusion Matrix](Graphs/{filename})\n\n")

def plot_roc(file, y_true, scores_dict, filename="roc_curve.png", title="ROC Curves"):
    plt.figure(figsize=(6, 7))
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    path = "Graphs/"
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close()
    file.write(f"![ROC Curve]({filepath})\n\n")

def plot_cm_avg(file, cm_avg, filename="cm_avg.png", title="Average Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["True 0","True 1"])
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    path = "Graphs/"
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close()
    file.write(f"![Average Confusion Matrix]({filepath})\n\n")

def plot_prob_distribution(file, y_prob, y_true, filename="prob_dist.png",
                           title="Predicted Probabilities by True Class"):
    plt.figure(figsize=(7, 5))
    plt.hist([y_prob[y_true == 0], y_prob[y_true == 1]], bins=20,
             label=["True 0", "True 1"], alpha=0.7, density=True)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()

    path = "Graphs/"
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close()
    file.write(f"![Probability Distribution]({filepath})\n\n")

def plot_feature_importance(file, coefs, feat_names, filename="feature_importance.png",
                            title="Feature Importance (Logistic Coefficients)"):
    sorted_idx = np.argsort(coefs)
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(feat_names)[sorted_idx], np.array(coefs)[sorted_idx])
    plt.xlabel("Coefficient value")
    plt.title(title)

    path = "Graphs/"
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close()
    file.write(f"![Feature Importance]({filepath})\n\n")

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
    feat_names = ["Clump Thickness",
                  "Uniformity of Cell Size",
                  "Uniformity of Cell Shape",
                  "Marginal Adhesion",
                  "Single Epithelial Cell Size",
                  "Bare Nuclei",
                  "Bland Chromatin",
                  "Normal Nucleoli",
                  "Mitosis",
                  ]

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
    # Best values for graphs
    manual_best_prob = None
    manual_best_coefs = None

    # -------------------- Scikit learn variables

    skl_acc_sum, skl_prec_sum, skl_rec_sum = 0, 0, 0
    skl_cms = []
    # [value, optimal act func, optimal iterations, optimal alpha range]
    skl_acc_best_data = [0, '', 0, 0]
    # [value, optimal act func, optimal iterations, optimal alpha range]
    skl_pres_best_data = [0, '', 0, 0]
    # [value, optimal act func, optimal iterations, optimal alpha range]
    skl_recall_best_data = [0, '', 0, 0]
    # Best values for graphs
    skl_best_prob = None
    skl_best_coefs = None



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
                # With the weighs get predicted output
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
                manual_acc_best_data, is_manual_new_best = get_best_values(
                    acc, manual_acc_best_data, name, iteration, curr_range_alpha)

                if is_manual_new_best:
                    manual_best_prob = predict_proba_manual(test_X, current_manual_weights, func)
                    manual_best_coefs = current_manual_weights


                # Compare best precision with current precision
                manual_pres_best_data, _ = get_best_values(
                    prec, manual_pres_best_data, name, iteration, curr_range_alpha)
                # Compare best recall with current recall
                manual_recall_best_data, _ = get_best_values(
                    rec, manual_recall_best_data, name, iteration, curr_range_alpha)

            # -------------------- Scikit learn testing
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    random_state=0, max_iter=iteration, C=curr_range_alpha, solver="lbfgs"
                ))
            ])
            # Train the model
            pipe.fit(training_X, training_y)
            # Predict the output
            y_pred = pipe.predict(test_X)
            y_prob = pipe.predict_proba(test_X)[:,1]
            # Calculate accuracy of model
            acc = accuracy_score(test_y, y_pred)
            # Calculate precision of model
            prec = precision_score(test_y, y_pred)
            # Calculate recall of model
            rec = recall_score(test_y, y_pred)
            # Calculate confusion matrix
            cm = confusion_matrix(test_y, y_pred)
            skl_cms.append(cm)
            # Sum predicts for the averages
            skl_acc_sum = skl_acc_sum + acc
            skl_prec_sum = skl_prec_sum + prec
            skl_rec_sum = skl_rec_sum + rec
            # Compare best accuracy with current accuracy
            skl_acc_best_data, is_skl_new_best = get_best_values(
                acc, skl_acc_best_data, '', iteration, curr_range_alpha)

            if is_skl_new_best:
                skl_best_prob = y_prob
                skl_best_coefs = pipe.named_steps["lr"].coef_[0]
            # Compare best precision with current precision
            skl_pres_best_data, _ = get_best_values(
                prec, skl_pres_best_data, '', iteration, curr_range_alpha)
            # Compare best recall with current recall
            skl_recall_best_data, _ = get_best_values(
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
        # Write manual results
        write_results(file, "Manual", manual_acc_best_data,
                      manual_pres_best_data, manual_recall_best_data,
                      manual_sig_avg_acc, manual_sig_avg_prec, manual_sig_avg_rec,
                      plot_confusion_matrix, manual_sig_avg_cm, manual_tanh_avg_acc,
                      manual_tanh_avg_prec, manual_tanh_avg_rec, manual_tanh_avg_cm)


        # -------------------- Manual Graphs
        file.write("### Manual Graphs\n\n")

        file.write(
            "For the **manual implementation**, we can see thorugh our graphs how the model "
            "behaves under its best configuration based on the highest accuracy. The ROC curve illustrates its ability to distinguish between classes "
            "across different thresholds, while the average confusion matrix shows the typical distribution of correct "
            "and incorrect predictions across multiple runs. The probability distribution plot helps us assess how confident "
            "the manual model is when making decisions, and the feature importance chart highlights which input variables "
            "had the greatest impact on its predictions.\n\n"
        )


        plot_roc(file=file,y_true=test_y,scores_dict={"Manual-best": manual_best_prob},filename="Manual_best_ROC.png",title="ROC - Manual (best)")

        file.write(
            "The ROC curve of the manual implementation exhibits a sharp increase in the true positive rate once the false positive rate "
            "exceeds 0.5, indicating that the model begins to capture a significant portion of the positives relatively quickly. "
            "After this point, the curve maintains a noticeable upward trend, continuing to gain sensitivity until it surpasses the 0.8 mark. "
            "Beyond this threshold, the growth becomes more gradual and the curve tends to flatten, moving closer to the diagonal slope. "
            "This pattern suggests that while the manual model is effective at identifying positive cases early on, its improvements slow "
            "down at higher thresholds, reflecting diminishing returns in discriminative capacity as it approaches its maximum performance.\n\n"
        )

        plot_cm_avg(file=file,cm_avg=manual_sig_avg_cm,filename="Manual_CM_Avg.png",title="Average Confusion Matrix - Manual")

        file.write(
            "The average confusion matrix of the manual implementation shows that the model is strong at recognizing negative cases, "
            "producing many true negatives and relatively few false positives. On the positive side, it is able to capture a significant "
            "portion of actual positive cases, although some false negatives remain. In the context of breast cancer detection, "
            "false negatives are particularly critical because they represent missed diagnoses. While the manual model demonstrates "
            "a balanced behavior, its performance suggests a slight tendency to prioritize avoiding false alarms over fully capturing "
            "all positive cases, which is an important consideration for medical applications.\n\n"
        )

        plot_prob_distribution(file=file,y_prob=manual_best_prob,y_true=test_y,filename="Manual_ProbDist.png",title="Predicted Probabilities - Manual (best)")

        file.write(
            "The probability distribution for the manual implementation shows that the model tends to assign probabilities "
            "close to 0 for negative cases and close to 1 for positive cases. This separation indicates that the classifier "
            "is confident in most of its predictions, with only a small number of instances falling into intermediate ranges. "
            "From a clinical perspective, such behavior is useful because it minimizes uncertainty when classifying a case: "
            "most predictions are made with high confidence. However, the few positive cases that appear near low probability "
            "values are concerning, as they represent situations where the model could miss a true cancer diagnosis. "
            "This underlines the importance of recall in medical applications, where it is preferable to reduce false negatives "
            "even if it comes at the cost of slightly more false positives.\n\n"
        )

        plot_feature_importance(file=file,coefs=np.array(manual_best_coefs),feat_names=feat_names,filename="Manual_FeatureImportance.png",title="Feature Importance - Manual (best)")

        file.write(
            "The feature importance plot for the manual implementation highlights which attributes of the dataset had the greatest impact "
            "on the model’s predictions. Variables such as *Bare Nuclei*, *Uniformity of Cell Size*, and *Normal Nucleoli* appear with strong "
            "positive coefficients, meaning that higher values in these features are strongly associated with predicting malignant cases. "
            "On the other hand, attributes like *Marginal Cell Size* and *Chromatin* show negative coefficients, suggesting that higher values "
            "of these variables push the prediction toward the benign class. Features closer to zero, such as *Clump Thickness* or *Cell Adhesion*, "
            "contributed little to the decision-making process in this configuration. In the context of breast cancer detection, this analysis "
            "helps to identify which cell characteristics the model found most informative for distinguishing between benign and malignant samples.\n\n"
        )

        # Write Scikit Learn results
        write_results(file, "Scikit learn", skl_acc_best_data,
                      skl_pres_best_data, skl_recall_best_data,
                      skl_avg_acc, skl_avg_prec, skl_avg_rec,
                      plot_confusion_matrix, skl_avg_cm)

        # -------------------- Manual Graphs
        file.write("### Scikit Learn Graphs\n\n")
        file.write(
            "For the **scikit-learn implementation**, we present the same set of graphs for consistency and comparison. "
            "The ROC curve reflects the classification trade-offs achieved by the optimized solver, and the average confusion "
            "matrix summarizes its stability over different runs. The probability distribution plot reveals how well-calibrated "
            "the scikit-learn model is in assigning probabilities, while the feature importance chart provides insights into "
            "how the learned coefficients influence the final predictions. Together, these figures allow us to contrast the "
            "manual and library-based approaches under a common framework.\n\n"
        )
        plot_roc(file=file,y_true=test_y,scores_dict={"Sklearn-best": skl_best_prob},filename="Sklearn_best_ROC.png",title="ROC - Scikit-learn (best)")

        file.write(
            "The ROC curve for the scikit-learn implementation shows a very strong discriminative performance. "
            "Unlike the manual approach, the curve begins at a higher true positive rate even when the false positive rate is close to zero, "
            "indicating that the model can correctly identify many positive cases from the start. "
            "The curve continues to rise quickly, reaching values close to the maximum true positive rate, and then stabilizes near the upper bound. "
            "This behavior suggests that the scikit-learn model achieves high sensitivity without incurring many false positives, "
            "and the area under the curve being close to 1 confirms that its overall classification power is superior compared to the manual implementation. "
            "In medical applications like breast cancer detection, this is a valuable property because it means the model can capture most positive cases "
            "with minimal risk of missing diagnoses.\n\n"
        )

        plot_cm_avg(file=file,cm_avg=skl_avg_cm,filename="Sklearn_CM_Avg.png",title="Average Confusion Matrix - Scikit-learn")

        file.write(
            "The average confusion matrix of the scikit-learn implementation reveals a very balanced performance, "
            "with a large majority of true negatives and true positives compared to the misclassified cases. "
            "The number of false positives is low, which means the model rarely classifies healthy patients as having cancer, "
            "and the number of false negatives is minimal, which is particularly important in a medical context. "
            "This pattern indicates that the scikit-learn model not only achieves strong precision by limiting false alarms, "
            "but also maintains a high recall by successfully detecting most positive cases. "
            "Overall, this configuration shows a model that is highly reliable for breast cancer detection, "
            "minimizing the risk of missed diagnoses while avoiding unnecessary false alerts.\n\n"
        )

        plot_prob_distribution(file=file,y_prob=skl_best_prob,y_true=test_y,filename="Sklearn_ProbDist.png",title="Predicted Probabilities - Scikit-learn (best)")

        file.write(
            "The probability distribution of the scikit-learn model shows a clearer separation between the two classes compared to the manual approach. "
            "Most negative cases concentrate near probability values close to zero, while the majority of positive cases are shifted toward values close to one. "
            "Although there is still some overlap in the intermediate range, the separation is more distinct, which indicates that the scikit-learn model assigns "
            "probabilities in a way that reflects stronger confidence in its predictions. From a medical standpoint, this behavior reduces uncertainty in decision-making, "
            "as the model provides more polarized outputs that make it easier to differentiate between benign and malignant cases. "
            "The reduced presence of positive cases in the low-probability region suggests a lower risk of false negatives compared to the manual implementation.\n\n"
        )

        plot_feature_importance(file=file,coefs=skl_best_coefs,feat_names=feat_names,filename="Sklearn_FeatureImportance.png",title="Feature Importance - Scikit-learn (best)")

        file.write(
            "The feature importance plot of the scikit-learn implementation shows that all coefficients are positive, "
            "which contrasts with the manual implementation where some variables contributed negatively. "
            "In this case, features such as *Uniformity of Cell Size*, *Bare Nuclei*, and *Cell Shape* stand out as the most influential, "
            "all pushing the prediction toward the malignant class when their values increase. "
            "Other attributes, including *Clump Thickness* and *Chromatin*, also show notable contributions, while *Mitosis* appears less relevant in comparison. "
            "The absence of negative coefficients suggests that, under the solver and scaling used by scikit-learn, the model interprets all features as risk indicators, "
            "though with varying magnitudes. This contrasts with the manual model, which allowed some variables to act as protective factors, "
            "reducing the likelihood of a malignant classification. From an interpretability standpoint, the scikit-learn results highlight a more uniform direction of influence, "
            "making it easier to identify which features consistently push the classification toward malignancy.\n\n"
        )

        write_conclusion(file)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    logistic_regression()
