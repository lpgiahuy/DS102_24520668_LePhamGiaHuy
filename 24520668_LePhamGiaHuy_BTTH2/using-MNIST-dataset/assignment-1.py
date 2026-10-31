import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
import struct

def load_mnist(filename, is_image=True):
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename

    with urllib.request.urlopen(url) as response:
        data = gzip.decompress(response.read())

    if is_image:
        _, n, rows, cols = struct.unpack(">IIII", data[:16])
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows * cols).copy()

    return np.frombuffer(data, dtype=np.uint8, offset=8).copy()

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=300):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0.0
        self.losses = []

        for epoch in range(self.epochs):
            predictions = self.predict_proba(X)
            errors = predictions - y

            gradient_weights = (X.T @ errors) / num_samples
            gradient_bias = np.mean(errors)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            # Binary Cross-Entropy Loss
            probabilities = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
            self.losses.append(loss)

def evaluate(y_true, y_pred):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for c in [0, 1]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Class {c}: Precision={precision:.4f}, "f"Recall={recall:.4f}, F1={f1:.4f}")

    print(f"\nModel Accuracy: {np.mean(y_true == y_pred):.4f}")
    print(f"Model Macro Precision: {np.mean(precision_scores):.4f}")
    print(f"Model Macro Recall: {np.mean(recall_scores):.4f}")
    print(f"Model Macro F1-score: {np.mean(f1_scores):.4f}")


def main():
    X_train = load_mnist("train-images-idx3-ubyte.gz")
    y_train = load_mnist("train-labels-idx1-ubyte.gz", False)

    X_test = load_mnist("t10k-images-idx3-ubyte.gz")
    y_test = load_mnist("t10k-labels-idx1-ubyte.gz", False)

    # Filter digits 0 and 1
    train_mask = (y_train == 0) | (y_train == 1)
    test_mask = (y_test == 0) | (y_test == 1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    # Normalize pixels to [0, 1]
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    model = LogisticRegression(learning_rate=0.1,epochs=300)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.plot(model.losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("MNIST Logistic Regression - Gradient Descent")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
