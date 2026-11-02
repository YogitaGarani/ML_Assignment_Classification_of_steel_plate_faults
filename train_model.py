# -*- coding: utf-8 -*-

import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# Load dataset
feature_cols = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
]
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
all_cols = feature_cols + target_cols


# Loaded from external URL at UCI ML Repo
url = 'http://apmonitor.com/pds/uploads/Main/steel.txt'

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

# Read and load the data
with urllib.request.urlopen(req) as response:
    data = pd.read_csv(response, sep=',', names=all_cols)


# Dataset information
dataset_name = "Steel Plates Faults"
dataset_source = "UCI Machine Learning Repository"
n_samples = data.shape[0]      # Total number of rows
n_features = data.shape[1] - len(target_cols)     # Number of features (excluding target)
problem_type = "multiclass_classification"

print(f"Dataset: {dataset_name}")
print(f"Source: {dataset_source}")
print(f"Samples: {n_samples}, Features: {n_features}")
print(f"Problem Type: {problem_type}")

print("Dataset description:\n")
print(data.describe())

# Preprocess the data
# Drop redundant header row
data = data.drop(index=0).reset_index(drop=True)

# One-hot encoding of the target values
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
y = np.argmax(data[target_cols].values, axis=1)


for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate features X and target y
data = data.dropna().reset_index(drop=True)
X = data.drop(columns=target_cols)

# Check for missing values
missing_counts = data.isnull().sum()
total_missing = missing_counts.sum()

print("=== Missing Value Check ===")
print(f"Total missing values in dataset: {total_missing}")

if total_missing > 0:
    print("\nColumns with missing values:")
    print(missing_counts[missing_counts > 0])
    print("Replacing all missing values with the median")
    X = X.fillna(X.median())
else:
    print("Clean dataset: No missing values found in any column.")


# 2. Train-Test Split and Standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test dataset to a separate CSV
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['Target'] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully!")

# Ensure model directory exists
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")

# 3. Model Configuration Dictionary
models = {
    "Logistic_Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),
    "Support_Vector_Machine": SVC(probability=True, random_state=42)
}

print("\n--- Model Training & Internal Validation Performance ---")
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)

    # Save the models to the model folder
    joblib.dump(clf, f"model/{name}.joblib")

    # Validation evaluations
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
    mcc = matthews_corrcoef(y_test, preds)

    print(f"\n[{name}]")
    print(f"  Accuracy:  {acc:.4f} | AUC: {auc:.4f}")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix Breakdown:")
    print(confusion_matrix(y_test, preds))

print("\nAll models trained and exported into the 'model/' directory successfully.")
