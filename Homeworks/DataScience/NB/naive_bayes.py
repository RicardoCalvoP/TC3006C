# Ricardo Calvo Perez - A01028889

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import warnings
import os



# -------------------- Manual NV functions

def manual_training(X, labels):
  data_size = len(X)
  num_features = X.shape[1]

  number_positive=len([x for x in labels if x == "positive"])
  number_negative=len([x for x in labels if x == "negative"])
  number_neutral=len([x for x in labels if x == "neutral"])

  p_probability = number_positive/float(data_size)
  neg_probability = number_negative/float(data_size)
  neu_probability = number_neutral/float(data_size)

  p_numerator_probability = [1] * num_features
  neg_numerator_probability = [1] * num_features
  neu_numerator_probability = [1] * num_features

  p_denominator_probability = 2
  neg_denominator_probability = 2
  neu_denominator_probability = 2
  #Iterate over training documents
  for x in range(data_size):
    #If is a vector with a positive label then
    if labels[x] == "positive":
      counter = 0
      #For all features in the training vector
      for y in X[x]:
        #if a feature appears in the vector (1) then increment the count
        p_numerator_probability[counter]+=y
        counter+=1
        #increment the total count for words associated to a positive label
        p_denominator_probability+=sum(X[x])
        #If is a vector with a nagative label then
    elif labels[x] == "negative":
      counter=0
      #For all features in the training vector
      for y in X[x]:
        #if a feature appears in the vector (1) then increment the count
        neg_numerator_probability[counter]+=y
        counter+=1
        #increment the total count for words associated to a negative label
        neg_denominator_probability+=sum(X[x])
        #Divide every feature/words by the total number of words for that class
    else:
      counter=0
      #For all features in the training vector
      for y in X[x]:
        #if a feature appears in the vector (1) then increment the count
        neu_numerator_probability[counter]+=y
        counter+=1
        #increment the total count for words associated to a negative label
        neu_denominator_probability+=sum(X[x])
        #Divide every feature/words by the total number of words for that class

  #Here we calculate the conditional probabilities for each p(xi|positive)
  pWordProbability=[]
  #For all features in the training vector
  for x in p_numerator_probability:
    pWordProbability.append(np.log(x/float(p_denominator_probability)))

  #Here we calculate the conditional probabilities for each p(xi|negative)
  negWordProbability=[]
  #For all features in the training vector
  for x in neg_numerator_probability:
    negWordProbability.append(np.log(x/float(neg_denominator_probability)))

  neuWordProbability=[]
  #For all features in the training vector
  for x in neu_numerator_probability:
    neuWordProbability.append(np.log(x/float(neu_denominator_probability)))

  #Return probabilities and conditional probabilities
  return (pWordProbability,negWordProbability, neuWordProbability, p_probability, neg_probability, neu_probability)


def classyfy_NB(X, p_word_probability, neg_word_probability, neu_word_probability, p_probability, neg_probability, neu_probability):
  counter = 0
  probabilities = []
  for x in X:
    probabilities.append(X[counter] * neg_word_probability[counter])
    counter += 1

  p0 = sum(probabilities) + np.log(neg_probability)

  counter = 0
  probabilities = []
  for x in X:
    probabilities.append(X[counter] * p_word_probability[counter])
    counter += 1

  p1 = sum(probabilities) + np.log(p_probability)

  counter = 0
  probabilities = []
  for x in X:
    probabilities.append(X[counter] * neu_word_probability[counter])
    counter += 1

  p2 = sum(probabilities) + np.log(neu_probability)

  if p0 > p1 and p0 > p2:
    return "negative"
  elif p1 > p0 and p1 > p2:
    return "positive"
  else:
    return "neutral"


# -------------------- Helper functions
def build_vocab(file_path, useless_words, num_features=10):
    vocab_counter = Counter()
    with open(file_path, "r", encoding="UTF-8") as file:
        for line in file:
            text, _label = line.split("@@@")
            tokens = text.strip().split()
            vocab_counter.update(tokens)
    # top-N por frecuencia
    vocab = []

    for w, _ in vocab_counter.most_common():
      if w not in useless_words:
        vocab.append(w)
        if len(vocab) == num_features:
          break

    return vocab

# -------------------- File functions

def read_file(file_path, vocab):
  idx = {w:i for i,w in enumerate(vocab)}
  X, y = [], []
  with open(file_path, "r", encoding="UTF-8") as file:
    for line in file:
        text, label = line.split("@@@")
        toks = text.strip().split()
        row = np.zeros(len(vocab), dtype=int)
        for t,c in Counter(toks).items():
            if t in idx: row[idx[t]] = c
        X.append(row)
        y.append(label.strip().lower())
    return np.array(X), np.array(y)

def read_raw(file_path):
  texts, labels = [], []
  with open(file_path, "r", encoding="utf-8") as f:
      for line in f:
          text, label = line.split("@@@")
          texts.append(text.strip())
          labels.append(label.strip().lower())
  return texts, labels

def set_index(file):
    file.write(
        "# Activity 3 Naive Bayes  with Scikit learn & manual\n\n")  # Title
    file.write("**Ricardo Calvo - A01028889**\n\n")  # Author
    file.write("## Table of Contents\n\n")  # Subtitle

    # Introduction
    file.write("1. [Introduction](#introduction)\n")  # Subtitle
    # Manual NB subtitles
    file.write("1. [Manual NB](#manual-nb)\n")  # Subtitle
    # Sklearn NB subtitles
    # Subtitle
    file.write("1. [Scikit learn NB](#scikit-learn-nb)\n")

    file.write("1. [Graphs](#graphs)\n")

    file.write("1. [Conclusion](#conclusion)\n")  # Subtitle


def write_introduction(file):
    file.write("## Introduction\n\n")
    file.write(
        "This report studies a Naive Bayes text classifier implemented in two ways: "
        "a manual Multinomial NB and a scikit-learn baseline. The task is 3-class sentiment "
        "classification with labels this ones being: positive, negative and neutral.\n\n"
    )
    file.write(
        "The dataset used is plain-text lines in the form `words @@@ label` split into training and test files. "
        "We convert text into a Bag-of-Words representation using tokenization, lowercasing, "
        "stopword filtering, and word frequency counts (not binary presence).\n\n"
        "To decide what words we didn't want in our model, we chose the F most popular words, "
        "and based on what words was, we discard it or not, for example, some of the words we dont want "
        "in our model are `and`, `or`, `is`, etc."

    )
    file.write(
        "For our manual model we did a multinomial Naive Bayes. We estimate class priors "
        "and per-class word likelihoods from the training data and score documents by the sum of "
        "log-likelihoods plus log-priors.\n\n"
    )
    file.write(
        "For the scikit-learn implementation we used a pipeline CountVectorizer → MultinomialNB. The vectorizer learns "
        "the vocabulary on each fold, enforces a maximum vocabulary size, and applies the same text "
        "preprocessing. The NB uses the default additive smoothing.\n\n"
    )
    file.write(
        "Evaluation: we sweep the vocabulary size F ∈ {20, 40, 60, 80, 100, 120} and perform "
        "Stratified K-fold cross-validation with K ∈ {3, 4, 5, 6} on the training split. "
        "We report macro accuracy, macro recall, and macro F1. We also show matrices that shows us "
        "a better view of how this network works. \n\n"
        "Finally, we compare the manual model against "
        "scikit-learn and discuss the impact of vocabulary size, stopwords, and class imbalance.\n\n"
    )

    file.write("[Return to Table of Contents](#table-of-contents)\n\n---\n\n")

def write_results(file, impl_name, best_acc, best_rec, best_f1):
    file.write(f"## {impl_name[0]} LR\n\n")

    file.write(
        f"Here are the best results we got with {impl_name[0]}: "
        f"**accuracy {best_acc[0][0]*100:.2f}%** with **{best_acc[0][1]} features**, a **recall "
        f"{best_rec[0][0]*100:.2f}%** with **{best_rec[0][1]} features**, "
        f"and a **F1 {best_f1[0][0]*100:.2f}%** with **{best_f1[0][1]} features**.\n\n"
    )
    file.write(
        "These results are on the low side, more likely because of a small or noisy vocabulary, "
        "generic words left in (or useful terms filtered out), simple tokenization, "
        "class imbalance that favors the majority class, weak handling of "
        "negations, or a train/test shift. We can improve by refining the stopword list "
        ", increasing the number of "
        "features as we can see that our best values, are with a greater number of features, "
        "and using stratified splits for evaluation.\n\n"
    )

    file.write(f"## {impl_name[1]} LR\n\n")

    file.write(
        f"For the best results using {impl_name[1]} we got an "
        f"**accuracy {best_acc[1][0]*100:.2f}%** with **{best_acc[1][1]} features** when "
        f"**K={best_acc[1][2]}**"
        f"a **recall {best_rec[1][0]*100:.2f}%** with **{best_rec[1][1]} features** when **K={best_rec[1][2]}**, "
        f"and an **F1 {best_f1[1][0]*100:.2f}%** with **{best_f1[1][1]} features** when **K={best_f1[1][2]}**.\n\n"
    )
    file.write(
        "These results reflect the pipeline setup that learns its own vocabulary during cross-validation "
        "and keeps preprocessing consistent. This scores feel low but they are greater than our manual implementation," \
        " trying a larger vocabulary, refine the "
        "stopword list, add lemmatization, may improve the check class balance.\n\n"
    )

    file.write("[Return to Table of Contents](#table-of-contents)\n\n---\n\n")

def write_conclusion(file):
    file.write("## Conclusion\n\n")
    file.write(
        "After doing the experiments we learned that vocabulary quality and preprocessing drive performance more than the specific Naive Bayes variant. "
        "As we increased the number of features, both models improved, but gains tapered off at larger vocabularies. "
        "We also saw a strong class imbalance toward the positive label, which pushed both models to over-predict that class and miss many negative and neutral cases. "
        "It is very important to filter what words you consider wont contribute as well as other words, normally this words are connection words as we talked in the introduction\n\n"
    )
    file.write(
        "Between the two approaches, the scikit-learn pipeline performed better overall. "
        "It consistently reached higher accuracy and competitive F1 because it learns the vocabulary within each split and applies preprocessing in a stable way. "
        "The manual model benefitted from more features but still trailed, suggesting that stronger text processing—refined stopwords while keeping negations, lemmatization, and n-grams—would help more than tweaks to the classifier itself.\n\n"
    )

    file.write("[Return to Table of Contents](#table-of-contents)\n\n---\n\n")



# -------------------- Graph functions

def plot_manual_vs_scikit(file, features, manual, sklearn, title):
    features = np.asarray(features)
    manual = np.asarray(manual, dtype=float)
    if features.shape[0] != manual.shape[0]:
        raise ValueError("features y acc_manual deben tener la misma longitud")

    # ordena por #features por si vienen desordenados
    order = np.argsort(features)
    f = features[order]
    am = manual[order]

    plt.figure()
    plt.plot(f, am, marker="o", linewidth=1.5, label="Manual")
    sklearn = np.asarray(sklearn, dtype=float)
    if sklearn.shape[0] != features.shape[0]:
        raise ValueError("features y acc_sklearn deben tener la misma longitud")
    plt.plot(f, sklearn[order], marker="o", linewidth=1.5, label="Scikit-learn")
    plt.legend()

    full_title = title + "_vs_Features"
    plt.title(full_title)
    plt.xlabel("#Features")
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    path = "Homeworks/DataScience/NB/Graphs/"
    os.makedirs(path, exist_ok=True)   # asegúrate que la carpeta exista
    filename = f"{full_title}.png"
    filepath = os.path.join(path, filename)

    # Si ya existe el archivo, lo elimina
    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    file.write(f"![{filename}](Graphs/{filename})\n\n")

def plot_confusion_best_manual(file, train_path, test_path, useless_words, best_F):
    vocab = build_vocab(train_path, useless_words, best_F)
    Xtr,ytr = read_file(train_path, vocab); Xte,yte = read_file(test_path, vocab)
    model = manual_training(Xtr, ytr)
    yhat = [classyfy_NB(x, *model) for x in Xte]
    cm = confusion_matrix(yte, yhat, labels=["positive","negative","neutral"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["pos","neg","neu"])
    plt.figure(); disp.plot(values_format="d", cmap="Blues"); plt.title(f"Manual Confusion (F={best_F})")
    fn="Manual_Confusion.png"; fp=os.path.join("Homeworks/DataScience/NB/Graphs", fn)
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    file.write(f"![{fn}](Graphs/{fn})\n\n")

def plot_confusion_best_sklearn(file, train_path, test_path, useless_words, best_F):
    texts_tr, ytr = read_raw(train_path); texts_te, yte = read_raw(test_path)
    pipe = make_pipeline(CountVectorizer(max_features=best_F, stop_words=sorted(useless_words)), MultinomialNB())
    pipe.fit(texts_tr, ytr); yhat = pipe.predict(texts_te)
    cm = confusion_matrix(yte, yhat, labels=["positive","negative","neutral"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["pos","neg","neu"])
    plt.figure(); disp.plot(values_format="d", cmap="Blues"); plt.title(f"Scikit Confusion (F={best_F})")
    fn="Scikit_Confusion.png"; fp=os.path.join("Homeworks/DataScience/NB/Graphs", fn)
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    file.write(f"![{fn}](Graphs/{fn})\n\n")

def plot_class_distribution(file, y_train, y_test=None, title="Class Distribution"):
    labels = ["positive","negative","neutral"]
    tr = [Counter(y_train)[l] for l in labels]
    te = [Counter(y_test)[l] for l in labels] if y_test is not None else None

    x = np.arange(len(labels)); w = 0.35
    plt.figure()
    plt.bar(x - (w/2 if te else 0), tr, width=w, label="Train")
    if te:
        plt.bar(x + w/2, te, width=w, label="Test")
    plt.xticks(x, labels); plt.ylabel("Count"); plt.title(title)
    plt.legend(); plt.tight_layout()

    save_dir = "Homeworks/DataScience/NB/Graphs"; os.makedirs(save_dir, exist_ok=True)
    fn = "Class_Distribution.png"; fp = os.path.join(save_dir, fn)
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    file.write(f"![{title}](Graphs/{fn})\n\n")

def plot_priors_vs_distribution(file, y_train, priors, title="Priors vs Class Share"):
    labels = ["positive","negative","neutral"]
    share = np.array([ (y_train==l).sum() for l in labels ], float); share /= share.sum()
    plt.figure()
    x = np.arange(len(labels)); w=0.35
    plt.bar(x-w/2, share, width=w, label="Data share")
    plt.bar(x+w/2, priors, width=w, label="NB priors")
    plt.xticks(x, labels); plt.ylim(0,1); plt.legend(); plt.title(title); plt.ylabel("Proportion")
    os.makedirs("Homeworks/DataScience/NB/Graphs", exist_ok=True)
    fn="Priors_vs_Share.png"; fp=os.path.join("Homeworks/DataScience/NB/Graphs", fn)
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    file.write(f"![{title}](Graphs/{fn})\n\n")

def plot_top_vocab_words(file, vocab, counts, top=20, title=" Most used words (in vocab)"):
    # vocab: list[str], counts: array-like (mismos índices)
    counts = np.asarray(counts, dtype=float)
    if len(vocab) != counts.shape[0]:
        raise ValueError("vocab y counts deben tener la misma longitud")

    order = np.argsort(-counts)[:top]
    words = [vocab[i] for i in order]
    vals  = counts[order]

    plt.figure()
    plt.barh(words[::-1], vals[::-1])
    plt.gca().invert_yaxis()
    plt.xlabel("Count")
    plt.title(str(top) + title)
    plt.tight_layout()

    save_dir = "Homeworks/DataScience/NB/Graphs"
    os.makedirs(save_dir, exist_ok=True)
    fn="Top_words.png"; fp=os.path.join("Homeworks/DataScience/NB/Graphs", fn)
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    file.write(f"![{title}](Graphs/{fn})\n\n")

# -------------------- Main functions

def naive_bayes():
  training_file_path = "Homeworks/DataScience/NB/training.txt"
  test_file_path = "Homeworks/DataScience/NB/test.txt"

  num_features_list = [ 40, 60, 80, 100, 120]

  words = [
    "the","to","in","on","a","and","i","of","for","is","at","with","be","you","it","my","that",
    "are","we","up","from","1st","get","time","if","he","about","go","as","one","last","now",
    "your","by","what","his","can","come","2","or","2nd","do","wait","an","they","watch",
    "back","got","after","know","show","when","has","make","more","some","there","still","think",
    "us","then","3rd","off","3","how","only", "sat","first", "1", "its", "im", "so", "u", "him", "ill",
    "them"
]

  useless_words = set(words)

  implementations = ["Manual", "Scikit Learn"]

  m_best_acc = [0, 0]
  m_best_recall = [0, 0]
  m_best_f1 = [0, 0]

  m_acc_history = []
  m_f1_history = []
  m_recall_history = []

  sl_best_acc = [0, 0, 0]
  sl_best_f1 = [0, 0, 0]
  sl_best_recall = [0, 0, 0]

  sl_acc_history = []
  sl_f1_history = []
  sl_recall_history = []


  for num_features in num_features_list:
    print(f"Num features: {num_features}")
    vocab = build_vocab(training_file_path, useless_words, num_features)
    X_training, y_training = read_file(training_file_path, vocab)
    X_test, y_test = read_file(test_file_path, vocab)

    # -------------------- Manual NV functions

    model = manual_training (X_training ,y_training)
    y_pred = []
    for vec, label in zip(X_test, y_test):
      predictedLabel=classyfy_NB(vec,model[0],model[1],model[2],model[3], model[4], model[5])
      y_pred.append(predictedLabel)
    #Calculates model accuracy
    acc = accuracy_score(y_test, y_pred)
    m_acc_history.append(acc)
    if acc > m_best_acc[0]:
       m_best_acc[0] = acc
       m_best_acc[1] = num_features

    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    m_recall_history.append(rec)
    if rec > m_best_recall[0]:
       m_best_recall[0] = rec
       m_best_recall[1] = num_features

    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    m_f1_history.append(f1)
    if f1 > m_best_f1[0]:
       m_best_f1[0] = f1
       m_best_f1[1] = num_features

    # -------------------- Scikit Learn NV functions

    texts_train, labels_train = read_raw(training_file_path)
    pipe = make_pipeline(
    CountVectorizer(max_features=num_features, stop_words=sorted(useless_words)),
    MultinomialNB(alpha=0.025)
    )

    for k in range(3, 7):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        acc = cross_val_score(pipe, texts_train, labels_train, cv=skf, scoring="accuracy").mean()
        if acc > sl_best_acc[0]:
          sl_best_acc[0] = acc
          sl_best_acc[1] = num_features
          sl_best_acc[2] = k
        rec = cross_val_score(pipe, texts_train, labels_train, cv=skf, scoring="recall_macro").mean()
        if rec > sl_best_recall[0]:
          sl_best_recall[0] = rec
          sl_best_recall[1] = num_features
          sl_best_recall[2] = k
        f1  = cross_val_score(pipe, texts_train, labels_train, cv=skf, scoring="f1_macro").mean()
        if f1 > sl_best_f1[0]:
          sl_best_f1[0] = f1
          sl_best_f1[1] = num_features
          sl_best_f1[2] = k

        if k == 6: sl_acc_history.append(acc); sl_recall_history.append(rec); sl_f1_history.append(f1)

  best_acc = [m_best_acc, sl_best_acc]
  best_rec = [m_best_recall, sl_best_recall]
  best_f1 = [m_best_f1, sl_best_f1]

  # -------------------- Create report
  filepath = "Homeworks/DataScience/NB/naive_bayes_a01028889.md"
  # If file exists, remove it
  if os.path.exists(filepath):
      os.remove(filepath)

  # Write results to the file
  with open(filepath, "w", encoding="UTF-8") as file:
      set_index(file)
      write_introduction(file)
      write_results(file, implementations, best_acc, best_rec, best_f1)

      file.write("## Graphs\n\n")
      file.write(
                  "The following graphs show how accuracy, recall, and F1 behaves with the number of features, "
                  "comparing the manual implementation with the scikit-learn implementation.\n\n"
                )

      plot_manual_vs_scikit(file, num_features_list, m_acc_history, sl_acc_history, "Accuracy")
      file.write(
          "### Accuracy vs. Number of Features\n\n"
          "The scikit-learn implementation shows higher accuracy across all vocabulary sizes. "
          "In both approaches, adding more features generally makes our accuracy higher. "
          "Scikit-learn decreases slightly with very small vocabularies and then climbs, and  the manual version rises more between mid and large vocabularies. "
          "The persistent gap suggests that fold-wise vocabulary learning and preprocessing help the scikit-learn pipeline capture more signal. "
          "Meaning better text preprocessing and feature extraction matter more than the classifier path. "
          "To improve the manual model, refine the stopword list while keeping negations, "
          "and increase the number of features until the curve stabilize.\n\n"
      )

      plot_manual_vs_scikit(file, num_features_list, m_recall_history, sl_recall_history, "Recall")
      file.write(
          "### Recall vs. Number of Features\n\n"
          "As we can see, the Recall increases as we add more features in both implementations simillar to the previous graph. The manual model stays ahead at all sizes, "
          "with a noticeable jump from mid to larger vocabularies. Scikit-learn also improves but more slowly, simillar to a linear growth. "
          "What this means is that a larger vocabulary helps the models recover more true examples from each class. "
          "To increasethe recall, filter  the vocabulary (keeping informative terms and negations) "
          "and address class imbalance.\n\n"
      )


      plot_manual_vs_scikit(file, num_features_list, m_f1_history, sl_f1_history, "F1")
      file.write(
          "### F1 vs. Number of Features\n\n"
          "Same to the other ones F1 increases steadily as the vocabulary grows in both implementations. "
          "With fewer features, scikit-learn leads, then the manual model catches up at the largest size. "
          "Meaning that with  more features we can improve the balance between precision and recall, with diminishing returns near the end.\n\n "
      )

      plot_confusion_best_manual(
        file, training_file_path, test_file_path, useless_words, m_best_f1[1]
    )
      file.write(
          f"## Manual Confusion matrix (F={m_best_f1[1]})\n\n"
          "This matrix shows a strong bias toward the **positive** class. Most errors are neutral→positive and negative→positive, "
          "so positive has high recall but only moderate precision. In contrast, **negative** and **neutral** have low recall "
          "(many of their true examples are missed) and are frequently predicted as positive.\n\n"
          "This model over predicts positive, likely due to class imbalance and limited vocabulary signal. "
          "Whether false positives or false negatives are more problematic depends on the use case, but this pattern means "
          "the system will often label non-positive texts as positive and will miss many negative/neutral classes.\n\n"
      )

      # SCIKIT-LEARN
      plot_confusion_best_sklearn(
          file, training_file_path, test_file_path, useless_words, sl_best_f1[1]
      )

      file.write(
          f"## Scikit-learn  Confusion matrix (F={sl_best_f1[1]})\n\n"
          "Similar to the Manual Confusion matrix, this model biased toward the **positive** class. Most mistakes are neutral→positive and negative→positive, "
          "so positive recall is high while precision is only moderate. **Negative** and especially **neutral** have low recall "
          "(many of their true cases are missed and flipped to positive).\n\n"
          "This model will mark many non-positive texts as positive, which is risky if false positives are costly. "
      )

      plot_class_distribution(file, y_training, y_test)  # o y_training, y_test

      file.write(
          "## Class distribution\n\n"
          "The plot shows clear imbalance since the positive class is larger then neutral and  negative. "
          "This pushes the learned priors toward positive, so with weak evidence the model tends to predict positive. "
          "This may affect theaccuracy making it look optimistic, recall for negative and neutral drops, and many of their true cases are mislabeled as positive.\n\n"
          "To mitigate this problems we can always show per-class precision, recall, and F1, and  try uniform or tuned class priors "
          "rebalance the training set and improve text preprocessing.\n\n"
      )
      print(m_best_acc[1])
      vocab = build_vocab(training_file_path, useless_words, m_best_acc[1])
      print(vocab)
      Xtr, ytr = read_file(training_file_path, vocab)  # (n_docs, |vocab|)
      counts = Xtr.sum(axis=0)                         # conteo total por palabra
      plot_top_vocab_words(file, vocab, counts, top=len(vocab))

      file.write(
          f"### Most frequent words\n\n"
          "Most of the top words carry little or no sentiment. They show up in all three classes, "
          "so they don’t separate them well. This isn’t necessarily bias, but it could represent weakness: "
          "in the model, leaning more on priors and whatever context slips through."
      )



      file.write("[Return to Table of Contents](#table-of-contents)\n\n---\n\n")

      write_conclusion(file)


if __name__ == "__main__":
  warnings.filterwarnings("ignore", message="The least populated class.*", category=UserWarning)
  naive_bayes()
  print("NB Success!")