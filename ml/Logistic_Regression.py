"""
Inputs: affected_html_elements + supplementary_information (raw text)
Output: wcag_reference (multi-class)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/Access_to_Tech_Dataset_cleaned.csv")

# combine text features
df["text"] = (
    df["affected_html_elements"].fillna("").astype(str)
    + " "
    + df["supplementary_information"].fillna("").astype(str)
)

X = df["text"]
y = df["wcag_reference"]

# remove classes with <7 samples
min_samples = 7
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
mask = y.isin(valid_classes)
X = X[mask]
y = y[mask]
print(f"Using {len(X)} samples across {y.nunique()} WCAG references (filtered from {len(df)} total)")

# splitting data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# TF-IDF vectorization on raw text
vectorizer = TfidfVectorizer(max_features=10_000, min_df=2, max_df=0.95, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# train classifier
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_vec, y_train)

# training set eval
y_train_pred = model.predict(X_train_vec)
print("\n--- Training ---")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(classification_report(y_train, y_train_pred, zero_division=0))

# validation set eval
y_val_pred = model.predict(X_val_vec)
print("\n--- Validation ---")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(classification_report(y_val, y_val_pred, zero_division=0))

# test set eval
y_test_pred = model.predict(X_test_vec)
print("\n--- Test ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, zero_division=0))
