import os
import numpy as np
import math
from statistics import mode
from operator import itemgetter

def euclideanDistance(list1,list2):
    sum_list = 0
    for x,y in zip(list1,list2):
        sum_list += ((y-x)**2)
    return math.sqrt(sum_list)

def classify(test_list, training_lists,training_labels, k):
    distance = []
    for training_list, label in zip (training_lists, training_labels):
        value = euclideanDistance(test_list,training_list)
        distance.append((value, label))

    distance.sort(key=itemgetter(0))
    vote_labels = []

    for x in distance[:k]:
        vote_labels.append(x[1])

    return mode(vote_labels)

def knn():
    """
    Generates and saves random data for a KNN classification task.
    """
    # Base path for saving the training data
    base_path = "./KNN_data/"
    # Number of data points
    num_data = 1000

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    # --- Training Data Generation ---
    print("Creating and saving training samples...")
    # Generate random data with the same size for features and labels
    train_feature1 = np.random.normal(0, 1, num_data)
    train_feature2 = np.random.normal(0, 1, num_data)
    train_label = np.random.choice(['A', 'B'], size=num_data, p=[0.5, 0.5])

    # Corrected open() function call with `encoding` argument
    with open(base_path + "KNNTraining.txt", "w", encoding="UTF-8") as file:
        for f1, f2, l in zip(train_feature1, train_feature2, train_label):
            file.write(f"{f1},{f2},{l}\n")

    # --- Test Data Generation ---
    print("Creating and saving test samples...")
    # Generate random data with the same size for features and labels
    test_feature1 = np.random.normal(0, 0.2, num_data)
    test_feature2 = np.random.normal(0, 0.2, num_data)
    test_label = np.random.choice(['A', 'B'], size=num_data, p=[0.5, 0.5])

    # Corrected open() function call with `encoding` argument
    with open(base_path + "KNNTest.txt", "w", encoding="UTF-8") as file:
        for f1, f2, l in zip(test_feature1, test_feature2, test_label):
            file.write(f"{f1},{f2},{l}\n")

    print("Data generation complete.")

    print("Apply the KNN approach over test samples")
    training=[]
    test=[]
    trainingLabels=[]
    testLabels = []
    k = 100

    print("Load training samples")
    with open(base_path + "KNNTest.txt", "r", encoding="UTF-8") as file:
      for line in file:
        elements=(line.rstrip('\n')).split(",")
        training.append([float(elements[0]),float(elements[1])])
        trainingLabels.append(elements[2])
    print("Load test samples")
    with open(base_path + "KNNTest.txt", "r", encoding="UTF-8") as file:
      for line in file:
        elements=(line.rstrip('\n')).split(",")
        test.append([float(elements[0]),float(elements[1])])
        testLabels.append(elements[2])
    #Apply the KNN approach over test samples using training data
    print("Apply the KNN approach over test samples")

    correctPredictions=0
    TotalPredictions=0
    count = 0
    for x,y in zip (test, testLabels):
      count +=1
      TotalPredictions=TotalPredictions+1
      predicted=classify(x, training, trainingLabels, k)
      if predicted==y:
        correctPredictions+=1
        print("Predicted (): "+str(predicted)+" realValue: "+str(y))

    print("Model accuracy: "+str((correctPredictions/TotalPredictions)*100)+"%")


if __name__ == "__main__":
    knn()