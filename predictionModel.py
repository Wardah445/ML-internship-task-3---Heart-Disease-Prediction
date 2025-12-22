import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("HeartDiseaseTrain-Test.csv")

print("Shape:", df.shape)
print(df.head())

print("\nColumn datatypes and missing counts:")
print(df.info())
print("\nMissing values per column:\n", df.isna().sum())

for c in df.columns:
    if df[c].dtype == object:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
print("\nMissing values after cleaning coercions:\n", df.isna().sum())
print("\nDataset description (numeric):")
print(df.describe().T)
print(df["target"].value_counts())

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Pairwise plots for a few important numeric features (to avoid too many plots)
sample_features = [c for c in num_cols if c != "target"][:6]
sns.pairplot(df[sample_features + ["target"]], hue="target")
plt.suptitle("Pairplot (sample features)", y=1.02)
plt.show()


# Boxplots to check outliers for numeric features
plt.figure(figsize=(12,6))
df[num_cols].boxplot(rot=45)
plt.title("Box plots of numeric features")
plt.show()

X = df.drop(columns=["target"])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# For numeric scaling:
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Apply preprocessor to training and test
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_proc, y_train)
y_pred_lr = lr.predict(X_test_proc)
y_proba_lr = lr.predict_proba(X_test_proc)[:, 1]

# Evaluation
def evaluate_model(y_test, y_pred, y_proba):
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(" Confusion Matrix")
    plt.show()
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    return acc, roc_auc

acc_lr, auc_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr)

