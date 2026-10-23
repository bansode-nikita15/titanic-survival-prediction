
"""
Titanic Survival Prediction
End-to-end ML pipeline using local Kaggle CSV files.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "titanic"
OUT = BASE_DIR / "report_assets"

OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# LOAD LOCAL DATASETS
# ------------------------------------------------------------

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

if not train_path.exists():
    raise FileNotFoundError(
        f"Training file not found: {train_path}\n"
        "Place train.csv inside the titanic folder."
    )

if not test_path.exists():
    raise FileNotFoundError(
        f"Test file not found: {test_path}\n"
        "Place test.csv inside the titanic folder."
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Standardize column names
train_df.columns = train_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()

# Validate expected Kaggle columns
required_train = {
    "passengerid", "survived", "pclass", "sex",
    "age", "sibsp", "parch", "fare", "embarked", "cabin"
}

required_test = required_train - {"survived"}

missing_train = required_train - set(train_df.columns)
missing_test = required_test - set(test_df.columns)

if missing_train:
    raise ValueError(
        f"Missing columns in train.csv: {sorted(missing_train)}"
    )

if missing_test:
    raise ValueError(
        f"Missing columns in test.csv: {sorted(missing_test)}"
    )

# Save passenger IDs for the final Kaggle submission
test_passenger_ids = test_df["passengerid"].copy()

# Work on independent copies
df = train_df.copy()
kaggle_test = test_df.copy()

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)
print("Training data shape:", df.shape)
print("Kaggle test data shape:", kaggle_test.shape)


# ------------------------------------------------------------
# BASIC DATA ANALYSIS
# ------------------------------------------------------------

info = {
    "train_shape": list(df.shape),
    "kaggle_test_shape": list(kaggle_test.shape),
    "survival_rate": round(float(df["survived"].mean() * 100), 2),
    "missing": {
        str(k): int(v)
        for k, v in df.isna().sum().items()
        if v > 0
    },
}

print("\nDataset information:")
print(json.dumps(info, indent=2))


# ------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

sns.countplot(x="survived", data=df, ax=ax[0, 0])
ax[0, 0].set_title("Survival Count (0 = Died, 1 = Survived)")

sns.barplot(x="sex", y="survived", data=df, ax=ax[0, 1], errorbar=None)
ax[0, 1].set_title("Survival Rate by Sex")

sns.barplot(x="pclass", y="survived", data=df, ax=ax[0, 2], errorbar=None)
ax[0, 2].set_title("Survival Rate by Passenger Class")

sns.histplot(
    data=df,
    x="age",
    hue="survived",
    bins=30,
    kde=True,
    ax=ax[1, 0],
)
ax[1, 0].set_title("Age Distribution by Survival")

sns.barplot(
    x="embarked",
    y="survived",
    data=df,
    errorbar=None,
    ax=ax[1, 1],
)
ax[1, 1].set_title("Survival Rate by Port")

sns.boxplot(x="survived", y="fare", data=df, ax=ax[1, 2])
ax[1, 2].set_yscale("log")
ax[1, 2].set_title("Fare by Survival (Log Scale)")

plt.tight_layout()
plt.savefig(OUT / "eda.png", dpi=130)
plt.close()


# Correlation matrix
numeric_df = df[
    ["survived", "pclass", "age", "sibsp", "parch", "fare"]
]

plt.figure(figsize=(7, 5))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUT / "corr.png", dpi=130)
plt.close()


# Group survival rates
groups = {
    "by_sex": (
        df.groupby("sex")["survived"].mean() * 100
    ).round(1).to_dict(),

    "by_class": (
        df.groupby("pclass")["survived"].mean() * 100
    ).round(1).to_dict(),

    "by_embarked": (
        df.groupby("embarked")["survived"].mean() * 100
    ).round(1).to_dict(),
}


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

for data in (df, kaggle_test):
    data["family_size"] = data["sibsp"] + data["parch"] + 1

    data["is_alone"] = (
        data["family_size"] == 1
    ).astype(int)

    data["has_cabin"] = (
        data["cabin"].notna()
    ).astype(int)

    data["deck"] = data["cabin"].str[0]


# Model input features
num_cols = [
    "age",
    "fare",
    "sibsp",
    "parch",
    "family_size",
    "pclass",
]

cat_cols = [
    "sex",
    "embarked",
]

bin_cols = [
    "is_alone",
    "has_cabin",
]

feature_cols = num_cols + cat_cols + bin_cols

X = df[feature_cols]
y = df["survived"].astype(int)

X_kaggle_test = kaggle_test[feature_cols]


# ------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

print("\nTraining split size:", len(X_train))
print("Validation split size:", len(X_test))
print("Unlabelled Kaggle test size:", len(X_kaggle_test))


# ------------------------------------------------------------
# PREPROCESSING PIPELINE
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            cat_cols,
        ),
        (
            "bin",
            "passthrough",
            bin_cols,
        ),
    ]
)


# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7,
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=SEED,
    ),

    "SVM (RBF)": SVC(
        probability=True,
        random_state=SEED,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=SEED,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

rows = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Cross-validation is performed only on the training split
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    rows.append({
        "Model": name,
        "CV Acc (mean)": cv_scores.mean(),
        "CV Acc (std)": cv_scores.std(),
        "Test Acc": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

results = (
    pd.DataFrame(rows)
    .sort_values("Test Acc", ascending=False)
    .reset_index(drop=True)
)

results.to_csv(OUT / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))


# ------------------------------------------------------------
# RANDOM FOREST HYPERPARAMETER TUNING
# ------------------------------------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(random_state=SEED),
    ),
])

param_grid = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

print("\nTuning Random Forest...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate on the held-out validation split
validation_predictions = best_model.predict(X_test)
validation_probabilities = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(
    y_test,
    validation_predictions,
    labels=[0, 1],
)

classification_rep = classification_report(
    y_test,
    validation_predictions,
    labels=[0, 1],
    target_names=["Died", "Survived"],
    zero_division=0,
)

tuned = {
    "best_params": {
        key.replace("classifier__", ""): value
        for key, value in grid_search.best_params_.items()
    },
    "cv_accuracy": float(grid_search.best_score_),
    "validation_accuracy": float(
        accuracy_score(y_test, validation_predictions)
    ),
    "precision": float(
        precision_score(y_test, validation_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, validation_predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, validation_predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, validation_probabilities)
    ),
}

print("\nTUNED RANDOM FOREST")
print(json.dumps(tuned, indent=2))
print("\nClassification Report:")
print(classification_rep)


# ------------------------------------------------------------
# MODEL COMPARISON PLOT
# ------------------------------------------------------------

plot_data = results.sort_values("Test Acc")

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    plot_data["Model"],
    plot_data["Test Acc"],
    color="#4C78A8",
)

for i, value in enumerate(plot_data["Test Acc"]):
    ax.text(value + 0.003, i, f"{value:.3f}", va="center")

ax.set_xlim(0, 1)
ax.set_xlabel("Validation Accuracy")
ax.set_title("Model Comparison")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=130)
plt.close()


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 4))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Died", "Survived"],
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)

ax.set_title("Tuned Random Forest - Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion.png", dpi=130)
plt.close()


# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    validation_probabilities,
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {tuned['roc_auc']:.3f})",
    color="#E45756",
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="grey",
    label="Random baseline",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc.png", dpi=130)
plt.close()


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

fitted_preprocessor = best_model.named_steps["preprocessor"]
fitted_classifier = best_model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
feature_names = [
    name.split("__")[-1] for name in feature_names
]

importance = pd.Series(
    fitted_classifier.feature_importances_,
    index=feature_names,
).sort_values()

plt.figure(figsize=(8, 5))
importance.plot.barh(color="#54A24B")
plt.title("Feature Importance - Tuned Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT / "importance.png", dpi=130)
plt.close()

top_importance = (
    importance.sort_values(ascending=False)
    .head(10)
    .round(4)
    .to_dict()
)


# ------------------------------------------------------------
# SAVE VALIDATION RESULTS
# ------------------------------------------------------------

summary = {
    "info": info,
    "groups": groups,
    "tuned_random_forest": tuned,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_rep,
    "top_importance": top_importance,
    "train_split_size": int(len(X_train)),
    "validation_split_size": int(len(X_test)),
    "kaggle_test_size": int(len(X_kaggle_test)),
    "model_comparison": results.to_dict(orient="records"),
}

with open(OUT / "results.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, default=str)


# ------------------------------------------------------------
# GENERATE KAGGLE TEST PREDICTIONS
# ------------------------------------------------------------

# The Kaggle test.csv has no Survived labels.
# Predictions here are for submission, not accuracy evaluation.

kaggle_predictions = best_model.predict(X_kaggle_test)

submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": kaggle_predictions.astype(int),
})

submission_path = BASE_DIR / "titanic_submission.csv"
submission.to_csv(submission_path, index=False)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nKaggle submission preview:")
print(submission.head(10).to_string(index=False))

print("\nSubmission saved to:", submission_path)
print("Reports and graphs saved to:", OUT)
print("\nGenerated files:")
for path in sorted(OUT.iterdir()):
    print(" -", path.name)
