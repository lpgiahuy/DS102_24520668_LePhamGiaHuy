import numpy as np
import urllib.request
import gzip
import struct

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score,classification_report)


def load_mnist(filename, is_image=True):
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename

    with urllib.request.urlopen(url) as response:
        data = gzip.decompress(response.read())

    if is_image:
        _, n, rows, cols = struct.unpack(">IIII", data[:16])
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows * cols).copy()

    return np.frombuffer(data, dtype=np.uint8, offset=8).copy()


def evaluate(y_true, y_pred):
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro F1-score : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    X_train = load_mnist("train-images-idx3-ubyte.gz")
    y_train = load_mnist("train-labels-idx1-ubyte.gz", False)

    X_test = load_mnist("t10k-images-idx3-ubyte.gz")
    y_test = load_mnist("t10k-labels-idx1-ubyte.gz", False)

    # Normalize pixels
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Logistic Regression (0 vs 1)
    train_mask = (y_train == 0) | (y_train == 1)
    test_mask = (y_test == 0) | (y_test == 1)

    X_train_binary = X_train[train_mask]
    y_train_binary = y_train[train_mask]
    X_test_binary = X_test[test_mask]
    y_test_binary = y_test[test_mask]

    logistic_model = LogisticRegression(solver="lbfgs", max_iter=300)

    logistic_model.fit(X_train_binary, y_train_binary)
    logistic_pred = logistic_model.predict(X_test_binary)

    print("\nLogistic Regression (Digits 0 and 1)")
    evaluate(y_test_binary, logistic_pred)

    # Softmax Regression (0 to 9)
    softmax_model = LogisticRegression(solver="lbfgs", max_iter=300)

    softmax_model.fit(X_train, y_train)
    softmax_pred = softmax_model.predict(X_test)

    print("\nSoftmax Regression (Digits 0 to 9)")
    evaluate(y_test, softmax_pred)

if __name__ == "__main__":
    main()
