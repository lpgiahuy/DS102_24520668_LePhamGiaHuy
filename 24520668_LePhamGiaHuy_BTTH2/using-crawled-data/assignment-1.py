#Predict whether an NHL team will achieve a winning percentage above 50% in the following season.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=2000):
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

        for _ in range(self.epochs):
            predictions = self.predict_proba(X)
            errors = predictions - y

            gradient_weights = (X.T @ errors) / num_samples
            gradient_bias = np.mean(errors)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

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
    df = pd.read_csv("../../24520668_LePhamGiaHuy_BTTH1/hockey_teams.csv")

    features = ["wins", "losses", "win_percent", "goals_for", "goals_against"]

    # Data cleaning
    df = df.dropna(subset=["team_name", "year"] + features).drop_duplicates(subset=["team_name", "year"])

    # Create next-season labels
    df = df.sort_values(["team_name", "year"]).copy()

    df["next_year"] = df.groupby("team_name")["year"].shift(-1)
    df["next_win"] = df.groupby("team_name")["win_percent"].shift(-1)

    df = df[df["next_year"] == df["year"] + 1].copy()
    df["target"] = (df["next_win"] > 0.5).astype(int)

    # Split train/test by year
    years = np.sort(df["year"].unique())
    split = int(len(years) * 0.8)

    train = df[df["year"].isin(years[:split])]
    test = df[df["year"].isin(years[split:])]

    X_train = train[features].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)

    X_test = test[features].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=int)

    # Standardization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print("Training samples:", len(y_train))
    print("Testing samples:", len(y_test))

    model = LogisticRegression(learning_rate=0.1,epochs=2000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(model.losses, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("NHL Logistic Regression - Gradient Descent")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
