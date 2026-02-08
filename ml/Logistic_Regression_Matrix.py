"""
Inputs: affected_html_elements + supplementary_information (raw text)
Output: 7 binary labels from Accessibility_Impact_Matrix.csv
"""

# loading libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, f1_score

# loading data
df = pd.read_csv("data/Access_to_Tech_Dataset_cleaned.csv")

# matrix 
matrix = pd.read_csv("data/Accessibility_Impact_Matrix.csv")

# merge dataset with matrix on violation_name
df = df.merge(
    matrix,
    left_on="violation_name",
    right_on="Violation Name",
    how="inner",
)
# categories to predict (matrix columns)
categories = ["Blind", "Low Vision", "Color Blind", "Motor", "Cognitive", "Auditory", "Situational"]
y_full = df[categories]

# combine text features (fill NaN with empty string)
df["text"] = (
    df["affected_html_elements"].fillna("").astype(str)
    + " "
    + df["supplementary_information"].fillna("").astype(str)
)
X = df["text"]

print(f"Using {len(X)} samples, predicting {len(categories)} impact categories")
print(f"Category distribution:\n{y_full.sum()}")

# splitting data (70, 15, 15)
# first separating training from the rest
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_full, test_size=0.30, random_state=42
)

# second split to divide validation and test evenly
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# TF-IDF vectorization on raw text
vectorizer = TfidfVectorizer(max_features=10_000, min_df=2, max_df=0.95, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# multi-label classifier with one logistic reg. per category
model = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
)
model.fit(X_train_vec, y_train)

# predictions
y_train_pred = model.predict(X_train_vec)
y_val_pred = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)

# model evaluations
results = {}
def eval_multilabel(y_true, y_pred, name):
    subset_acc = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    results[name] = {
        "Subset Accuracy": subset_acc,
        "Micro F1": micro_f1
    }

    print(f"\n{name}")
    print(f"Subset accuracy: {subset_acc:.3f}")
    print(f"Micro F1:        {micro_f1:.3f}")

eval_multilabel(y_train, y_train_pred, "Training")
eval_multilabel(y_val, y_val_pred, "Validation")
eval_multilabel(y_test, y_test_pred, "Test")

# modeling performance across data splits
'''
metrics_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(7, 4))
metrics_df.plot(kind='line', marker='o', ax=ax)
ax.set_title("Model Performance Across Data Splits")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(True)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
'''
