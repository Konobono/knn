import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv("vine_final.csv", sep=";", decimal=",")
# data.drop(data.columns[2], axis=1, inplace=True)
data = data.replace('?', np.NaN)
data = data.apply(pd.to_numeric)
data = shuffle(data)
data = data.sample(frac=1, random_state=42)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

n_samples = len(X)
n_train = int(0.6 * n_samples)
n_val = int(0.2 * n_samples)
n_test = n_samples - n_train - n_val

np.random.seed(42)
indices = np.random.permutation(n_samples)
X_train = X[indices[:n_train]]
y_train = y[indices[:n_train]]
X_val = X[indices[n_train:n_train + n_val]]
y_val = y[indices[n_train:n_train + n_val]]
X_test = X[indices[n_train + n_val:]]
y_test = y[indices[n_train + n_val:]]

k = 3



def euclidean_distance(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))


def predict_label(X_train, y_train, x_test, k):

    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return max(set(k_labels), key=k_labels.count)


def confusion_matrix(true_label, pred_label):

    num_classes = len(np.unique(true_label))
    con_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(true_label)):
        true_label_check = true_label[i]
        pred_label_check = pred_label[i]
        con_matrix[true_label_check-1][pred_label_check-1] += 1
    return con_matrix


def instances_count(con_matrix_check):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(con_matrix_check)):
        for j in range(len(con_matrix_check)):
            if i == j:
                tp += con_matrix_check[i][j]
            elif con_matrix_check[j][i] > 0:
                fp += con_matrix_check[j][i]
            elif con_matrix_check[i][j] > 0:
                fn += con_matrix_check[i][j]
            elif i != j:
                tn += sum(con_matrix_check[j]) - con_matrix_check[j][j]

    return tp, tn, fp, fn


best_k = None
best_accuracy = 0
for k in range(1, 21):
    y_pred_val = [predict_label(X_train, y_train, x_val, k) for x_val in X_val]
    accuracy = np.mean(y_val == y_pred_val)
    print("k:", k, "Accuracy:", accuracy*100, "%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

y_pred_test = [predict_label(X_train, y_train, x_test, best_k) for x_test in X_test]
accuracy = np.mean(y_test == y_pred_test)
print("Best k:", best_k, "Accuracy on testing data:", accuracy*100, "%")

tabela = confusion_matrix(y_test, y_pred_test)
print(tabela)
ile_tych_pomylek = instances_count(tabela)
print(ile_tych_pomylek)

# przeskalowac dane z win (duze i male wartosci)
