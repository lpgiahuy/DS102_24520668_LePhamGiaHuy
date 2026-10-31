# Predict NHL teams' next-season performance using Scikit-learn.
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score,classification_report)


def evaluate(y_true, y_pred):
    print(f"Model Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Model Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Model Macro Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Model Macro F1-score: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    print(classification_report(y_true, y_pred, zero_division=0))


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

    df["binary_target"] = (df["next_win"] > 0.5).astype(int)

    df["multiclass_target"] = np.select([df["next_win"] <= 0.4, df["next_win"] <= 0.5],[0, 1],default=2)

    # Split train/test by year
    years = np.sort(df["year"].unique())
    split = int(len(years) * 0.8)

    train = df[df["year"].isin(years[:split])]
    test = df[df["year"].isin(years[split:])]

    X_train = train[features].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)

    # Standardization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Logistic Regression (Binary Classification)
    y_train_binary = train["binary_target"].to_numpy()
    y_test_binary = test["binary_target"].to_numpy()

    logistic_model = LogisticRegression(solver="lbfgs",max_iter=2000)

    logistic_model.fit(X_train, y_train_binary)
    logistic_pred = logistic_model.predict(X_test)

    print("\nLogistic Regression (Binary Classification)")
    evaluate(y_test_binary, logistic_pred)

    # Softmax Regression (Multiclass Classification)
    y_train_multi = train["multiclass_target"].to_numpy()
    y_test_multi = test["multiclass_target"].to_numpy()

    softmax_model = LogisticRegression(solver="lbfgs",max_iter=2000)

    softmax_model.fit(X_train, y_train_multi)
    softmax_pred = softmax_model.predict(X_test)

    print("\nSoftmax Regression (Multiclass Classification)")
    evaluate(y_test_multi, softmax_pred)

if __name__ == "__main__":
    main()