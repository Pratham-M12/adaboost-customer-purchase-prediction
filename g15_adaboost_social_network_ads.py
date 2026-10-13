"""
ISE 2 - Group G15 | Ensemble Learning
Dataset      : Social Network Ads
Application  : Customer Purchase Prediction
Base models  : Decision-Tree-based learners
Ensemble     : Boosting - AdaBoost

Central question:
  Does combining weak tree learners with AdaBoost improve purchase prediction
  compared with individual tree-based models?
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# 0. SETTINGS (change only here)
# ----------------------------------------------------------------------
CSV_PATH = "Social_Network_Ads.csv"
FEATURES = ["Age", "EstimatedSalary"]   # input features  (X)
TARGET = "Purchased"                    # output / label  (y)
RANDOM_STATE = 42                       # reproducibility
TEST_SIZE = 0.20                        # 80 : 20 split
RUN_TUNING = True                       # optional GridSearch on TRAIN data only

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join("figures", name), dpi=200, bbox_inches="tight")
    plt.show()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# 1. LOAD + INSPECT THE DATASET
# ----------------------------------------------------------------------
section("1. LOAD AND INSPECT")

if not os.path.exists(CSV_PATH):
    sys.exit(f"Cannot find '{CSV_PATH}'. Download the Social Network Ads CSV "
             "and place it next to this script (or edit CSV_PATH).")

df = pd.read_csv(CSV_PATH)

missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    sys.exit(f"Column(s) {missing_cols} not found. Your columns are: "
             f"{list(df.columns)}. Edit FEATURES / TARGET at the top.")

print("First 5 rows:")
print(df.head(), "\n")
print("Shape (rows, columns):", df.shape)
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

n_dup_full = df.duplicated().sum()
n_dup_model = df.duplicated(subset=FEATURES + [TARGET]).sum()
print("\nDuplicate rows (all columns)            :", n_dup_full)
print("Duplicate rows (features + target only) :", n_dup_model)

print("\nSummary statistics (feature ranges):")
print(df[FEATURES + [TARGET]].describe().T)

print("\nTarget values present:", sorted(df[TARGET].unique()))
counts = df[TARGET].value_counts().sort_index()
percent = (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2)
print("\nClass distribution:")
print(pd.DataFrame({"count": counts, "percent": percent}))
print("Majority-class share (the accuracy of 'always predict the majority'): "
      f"{percent.max():.2f}%")

print("\nPotential outliers by the 1.5 x IQR rule:")
for col in FEATURES:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"  {col}: {n_out} values outside [{low:.1f}, {high:.1f}]")

# ----------------------------------------------------------------------
# 2. EXPLORATORY PLOTS
# ----------------------------------------------------------------------
section("2. EXPLORATORY PLOTS")

# 2a. Purchase distribution
plt.figure(figsize=(6, 5))
ax = sns.countplot(x=TARGET, data=df, palette=["#6c8ebf", "#e07b39"], hue=TARGET,
                   legend=False)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
ax.set_xticklabels(["0 = No purchase", "1 = Purchased"])
ax.set_xlabel("")
ax.set_title("Purchase distribution")
save_fig("01_purchase_distribution.png")

# 2b. Feature distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(data=df, x="Age", hue=TARGET, kde=True, ax=axes[0],
             palette=["#6c8ebf", "#e07b39"])
axes[0].set_title("Age distribution by class")
sns.histplot(data=df, x="EstimatedSalary", hue=TARGET, kde=True, ax=axes[1],
             palette=["#6c8ebf", "#e07b39"])
axes[1].set_title("Estimated salary distribution by class")
save_fig("02_feature_distributions.png")

# 2c. Scatter: Age vs Salary
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="EstimatedSalary", hue=TARGET,
                palette=["#6c8ebf", "#e07b39"], alpha=0.8)
plt.title("Age vs Estimated Salary (colour = Purchased)")
save_fig("03_age_salary_scatter.png")

# ----------------------------------------------------------------------
# 3. FEATURES / TARGET AND TRAIN-TEST SPLIT
# ----------------------------------------------------------------------
section("3. TRAIN-TEST SPLIT (80:20, stratified, fixed random_state)")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])
print("Class % in train:", (y_train.value_counts(normalize=True).sort_index() * 100).round(2).to_dict())
print("Class % in test :", (y_test.value_counts(normalize=True).sort_index() * 100).round(2).to_dict())
print(f"NOTE: with {len(y_test)} test samples, ONE sample changes accuracy by "
      f"{100 / len(y_test):.2f} percentage points. Small differences between "
      "models may be noise -> see the repeated cross-validation in section 9.")

# ----------------------------------------------------------------------
# 4. FEATURE SCALING (sanity check - trees do NOT need it)
# ----------------------------------------------------------------------
section("4. FEATURE SCALING SANITY CHECK")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on TRAIN only (no leakage)
X_test_sc = scaler.transform(X_test)         # reuse train mean/std on test

dt_raw = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
dt_scaled = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train_sc, y_train)
diff = (dt_raw.predict(X_test) != dt_scaled.predict(X_test_sc)).sum()
print("Test predictions that differ between raw-feature and scaled-feature "
      f"Decision Tree: {diff}")
print("Raw accuracy   :", round(accuracy_score(y_test, dt_raw.predict(X_test)), 4))
print("Scaled accuracy:", round(accuracy_score(y_test, dt_scaled.predict(X_test_sc)), 4))
print("Trees split on thresholds of ONE feature at a time, so a monotonic "
      "rescaling does not change which samples fall in which region.\n"
      "-> All main experiments below use the RAW (unscaled) features, which "
      "also keeps thresholds readable (e.g. Age <= 36.5).")


# ----------------------------------------------------------------------
# 5. DEFINE MODELS
# ----------------------------------------------------------------------
section("5. MODELS")


def make_adaboost(depth=1, n_estimators=50, learning_rate=1.0):
    """AdaBoost whose weak learner is a decision tree of the given depth.
    scikit-learn >= 1.2 uses `estimator=` (older `base_estimator=` is removed).
    scikit-learn >= 1.6 always uses the SAMME algorithm, so we only pass
    algorithm='SAMME' on older versions to get the same behaviour."""
    weak = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    kwargs = dict(estimator=weak, n_estimators=n_estimators,
                  learning_rate=learning_rate, random_state=RANDOM_STATE)
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) < (1, 6):
        kwargs["algorithm"] = "SAMME"
    return AdaBoostClassifier(**kwargs)


# name -> (model, group)
models = {
    "Majority baseline":             (DummyClassifier(strategy="most_frequent"), "Baseline"),
    "Decision Stump (depth 1)":      (DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE), "Individual tree"),
    "Shallow Tree (depth 3)":        (DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE), "Individual tree"),
    "Full Decision Tree":            (DecisionTreeClassifier(random_state=RANDOM_STATE), "Individual tree"),
    "Random Forest (bagging ref.)":  (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), "Bagging reference"),
    "AdaBoost (stumps, 50)":         (make_adaboost(depth=1, n_estimators=50), "Boosting"),
    "AdaBoost (depth-2 trees, 50)":  (make_adaboost(depth=2, n_estimators=50), "Boosting"),
}
for name, (_, grp) in models.items():
    print(f"  [{grp:<17}] {name}")

# ----------------------------------------------------------------------
# 6. TRAIN, PREDICT, EVALUATE (single 80:20 split)
# ----------------------------------------------------------------------
section("6. HOLD-OUT TEST RESULTS (single 80:20 split)")

rows, preds, probas, cms = [], {}, {}, {}
for name, (model, grp) in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    preds[name], probas[name] = y_pred, y_prob
    cms[name] = confusion_matrix(y_test, y_pred)
    rows.append({
        "Model": name,
        "Group": grp,
        "Train Acc": accuracy_score(y_train, model.predict(X_train)),
        "Test Acc": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })

results = pd.DataFrame(rows).set_index("Model")
print(results.round(4).to_string())
results.round(4).to_csv("results/holdout_results.csv")

print("\nConfusion matrices (rows = actual 0/1, columns = predicted 0/1):")
for name, cm in cms.items():
    tn, fp, fn, tp = cm.ravel()
    print(f"  {name:<30} TN={tn:<3} FP={fp:<3} FN={fn:<3} TP={tp:<3}")

print("\nDetailed report for AdaBoost (stumps):")
print(classification_report(y_test, preds["AdaBoost (stumps, 50)"],
                            target_names=["No purchase", "Purchased"], zero_division=0))

# ----------------------------------------------------------------------
# 7. PLOTS: confusion matrices, metric bars, ROC
# ----------------------------------------------------------------------
section("7. COMPARISON PLOTS")

plot_models = [m for m in models if m != "Majority baseline"]

# 7a. Confusion matrices
n = len(plot_models)
cols = 3
rws = int(np.ceil(n / cols))
fig, axes = plt.subplots(rws, cols, figsize=(5 * cols, 4.3 * rws))
for ax, name in zip(axes.ravel(), plot_models):
    sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
    ax.set_title(name, fontsize=12)
for ax in axes.ravel()[n:]:
    ax.axis("off")
save_fig("04_confusion_matrices.png")

# 7b. Accuracy comparison
order = results.loc[plot_models]
colors = {"Individual tree": "#6c8ebf", "Bagging reference": "#9aa5b1", "Boosting": "#e07b39"}
plt.figure(figsize=(11, 5.5))
bars = plt.bar(order.index, order["Test Acc"], color=[colors[g] for g in order["Group"]])
plt.ylim(max(0, order["Test Acc"].min() - 0.1), 1.0)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Test accuracy")
plt.title("Test accuracy: individual trees vs AdaBoost")
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
             f"{b.get_height():.3f}", ha="center", fontsize=11)
save_fig("05_accuracy_comparison.png")

# 7c. Precision / Recall / F1
order[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(12, 5.5), width=0.8)
plt.ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.title("Precision, Recall and F1 by model")
plt.legend(loc="lower right")
save_fig("06_precision_recall_f1.png")

# 7d. ROC curves
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name in plot_models:
    RocCurveDisplay.from_predictions(y_test, probas[name], name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_title("ROC curves (test set)")
ax.legend(fontsize=9, loc="lower right")
save_fig("07_roc_curves.png")

# ----------------------------------------------------------------------
# 8. DECISION BOUNDARIES + ADABOOST INTERNALS
# ----------------------------------------------------------------------
section("8. DECISION BOUNDARIES AND ADABOOST INTERNALS")

# 8a. Decision boundaries on raw Age / Salary
age_rng = np.linspace(X["Age"].min() - 2, X["Age"].max() + 2, 300)
sal_rng = np.linspace(X["EstimatedSalary"].min() - 5000, X["EstimatedSalary"].max() + 5000, 300)
xx, yy = np.meshgrid(age_rng, sal_rng)
grid = pd.DataFrame({FEATURES[0]: xx.ravel(), FEATURES[1]: yy.ravel()})

boundary_models = ["Decision Stump (depth 1)", "Full Decision Tree",
                   "Random Forest (bagging ref.)", "AdaBoost (stumps, 50)"]
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, name in zip(axes.ravel(), boundary_models):
    zz = models[name][0].predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_test["Age"], X_test["EstimatedSalary"], c=y_test, cmap="coolwarm",
               edgecolor="k", s=45)
    ax.set_title(f"{name}\n(points = TEST samples)", fontsize=12)
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated salary")
save_fig("08_decision_boundaries.png")

# 8b. AdaBoost: performance as learners are added (n_estimators effect)
ada = models["AdaBoost (stumps, 50)"][0]
stump = models["Decision Stump (depth 1)"][0]
train_curve = list(ada.staged_score(X_train, y_train))
test_curve = list(ada.staged_score(X_test, y_test))
rounds = np.arange(1, len(test_curve) + 1)

plt.figure(figsize=(9, 5.5))
plt.plot(rounds, train_curve, label="AdaBoost - train accuracy")
plt.plot(rounds, test_curve, label="AdaBoost - test accuracy")
plt.axhline(accuracy_score(y_test, stump.predict(X_test)), color="gray", ls="--",
            label="Single stump - test accuracy")
plt.xlabel("Number of weak learners combined")
plt.ylabel("Accuracy")
plt.title("AdaBoost (stumps): accuracy vs number of learners")
plt.legend()
save_fig("09_adaboost_accuracy_vs_learners.png")

# 8c. Learner errors and learner weights (alpha)
n_used = len(ada.estimators_)
errs = ada.estimator_errors_[:n_used]
alphas = ada.estimator_weights_[:n_used]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, n_used + 1), errs, marker="o", ms=3)
axes[0].axhline(0.5, color="r", ls="--", label="error = 0.5 (random guess)")
axes[0].set_title("Weighted error of each weak learner")
axes[0].set_xlabel("Boosting round t")
axes[0].set_ylabel("Weighted error")
axes[0].legend()
axes[1].bar(range(1, n_used + 1), alphas, color="#e07b39")
axes[1].set_title("Learner weight (alpha) of each weak learner")
axes[1].set_xlabel("Boosting round t")
axes[1].set_ylabel("alpha")
save_fig("10_adaboost_errors_and_alphas.png")

print("First 5 boosting rounds (AdaBoost with stumps):")
print(pd.DataFrame({"round": range(1, 6), "weighted_error": errs[:5].round(4),
                    "alpha (sklearn)": alphas[:5].round(4),
                    "ln((1-err)/err)": np.log((1 - errs[:5]) / errs[:5]).round(4)}).to_string(index=False))
print("For 2 classes scikit-learn's alpha equals learning_rate * ln((1-err)/err).\n"
      "The textbook formula has an extra factor 1/2; this only rescales all "
      "alphas equally, so the final weighted vote has the same sign.")

# 8d. Effect of learning_rate (illustration only - do NOT choose settings from test curves)
plt.figure(figsize=(9, 5.5))
for lr in [0.1, 0.5, 1.0]:
    m = make_adaboost(depth=1, n_estimators=200, learning_rate=lr).fit(X_train, y_train)
    plt.plot(range(1, len(list(m.estimators_)) + 1), list(m.staged_score(X_test, y_test)),
             label=f"learning_rate = {lr}")
plt.xlabel("Number of weak learners")
plt.ylabel("Test accuracy")
plt.title("Effect of n_estimators and learning_rate (illustration)")
plt.legend()
save_fig("11_learning_rate_effect.png")

# 8e. Visualise one small tree (viva-friendly)
plt.figure(figsize=(12, 6))
plot_tree(models["Shallow Tree (depth 3)"][0], feature_names=FEATURES,
          class_names=["No purchase", "Purchased"], filled=True, rounded=True, fontsize=9)
plt.title("Decision tree (max_depth = 3) trained on the training set")
save_fig("12_decision_tree_depth3.png")

# ----------------------------------------------------------------------
# 9. REPEATED STRATIFIED CROSS-VALIDATION (more reliable than one split)
# ----------------------------------------------------------------------
section("9. REPEATED STRATIFIED 5-FOLD CROSS-VALIDATION (5 repeats = 25 folds)")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_rows, cv_acc = [], {}
for name, (model, grp) in models.items():
    # cross_validate clones the model, so the fitted models above are untouched
    s = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_acc[name] = s["test_accuracy"]
    row = {"Model": name, "Group": grp}
    for m in scoring:
        row[m] = f"{s['test_' + m].mean():.3f} +/- {s['test_' + m].std():.3f}"
    cv_rows.append(row)

cv_table = pd.DataFrame(cv_rows).set_index("Model")
print(cv_table.to_string())
cv_table.to_csv("results/cv_results.csv")

plt.figure(figsize=(11, 5.5))
plt.boxplot([cv_acc[m] for m in plot_models], showmeans=True)
plt.xticks(range(1, len(plot_models) + 1), plot_models, rotation=25, ha="right")
plt.ylabel("Accuracy over 25 folds")
plt.title("Cross-validated accuracy (spread shows how stable each model is)")
save_fig("13_cv_accuracy_boxplot.png")

# ----------------------------------------------------------------------
# 10. OPTIONAL: TUNE ADABOOST ON THE TRAINING DATA ONLY
# ----------------------------------------------------------------------
if RUN_TUNING:
    section("10. OPTIONAL HYPERPARAMETER TUNING (training data only)")
    param_grid = {
        "estimator__max_depth": [1, 2, 3],
        "n_estimators": [25, 50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    }
    gs = GridSearchCV(
        make_adaboost(),
        param_grid, scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)          # test set is NOT touched while tuning
    print("Best parameters (by 5-fold CV F1 on train set):", gs.best_params_)
    print("Best CV F1 on train set:", round(gs.best_score_, 4))
    best = gs.best_estimator_
    yb = best.predict(X_test)
    print("\nTuned AdaBoost evaluated ONCE on the test set:")
    print("  Accuracy :", round(accuracy_score(y_test, yb), 4))
    print("  Precision:", round(precision_score(y_test, yb, zero_division=0), 4))
    print("  Recall   :", round(recall_score(y_test, yb, zero_division=0), 4))
    print("  F1       :", round(f1_score(y_test, yb, zero_division=0), 4))
    print("  ROC-AUC  :", round(roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]), 4))

# ----------------------------------------------------------------------
# 11. TEACHING ONLY: AdaBoost from scratch (explains the maths in code)
# ----------------------------------------------------------------------
section("11. ADABOOST FROM SCRATCH (teaching demo, 10 rounds)")

Xtr, Xte = X_train.values, X_test.values
ytr = np.where(y_train.values == 1, 1, -1)       # AdaBoost maths uses labels -1 / +1
yte = np.where(y_test.values == 1, 1, -1)

T = 10
w = np.full(len(ytr), 1 / len(ytr))              # Step 1: equal weights
learners, alphas_s = [], []
for t in range(1, T + 1):
    h = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    h.fit(Xtr, ytr, sample_weight=w)             # Step 2: train on weighted data
    pred = h.predict(Xtr)
    err = np.sum(w[pred != ytr])                 # Step 3: weighted error
    err = np.clip(err, 1e-10, 1 - 1e-10)
    if err >= 0.5:                               # worse than random -> stop
        print(f"Round {t}: weighted error {err:.3f} >= 0.5, stopping.")
        break
    alpha = 0.5 * np.log((1 - err) / err)        # Step 4: learner weight
    w = w * np.exp(-alpha * ytr * pred)          # Step 5: raise weight of mistakes
    w = w / w.sum()                              # Step 6: normalise
    learners.append(h)
    alphas_s.append(alpha)
    print(f"Round {t:>2}: weighted error = {err:.4f} | alpha = {alpha:.4f} | "
          f"max sample weight = {w.max():.4f}")

votes = sum(a * h.predict(Xte) for a, h in zip(alphas_s, learners))
final = np.where(votes >= 0, 1, -1)              # Step 7: sign of weighted vote
print("\nFrom-scratch AdaBoost test accuracy:", round(np.mean(final == yte), 4))
print("(Only to show the mechanism - use the scikit-learn model for reported results.)")

section("DONE - figures in ./figures, tables in ./results")
