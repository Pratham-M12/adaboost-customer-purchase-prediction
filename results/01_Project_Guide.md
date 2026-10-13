# ISE 2 — Group G15 — Project Guide (Part 1 of 4)

**AdaBoost on Social Network Ads: Customer Purchase Prediction**
Covers deliverables 1–14: project understanding, approach, dataset, problem statement, application, base models, ensemble theory, AdaBoost theory and maths, methodology, code pointer, metrics, results template, graphs.

**How to read the labels used throughout**
- **[THEORY]** — textbook fact, true regardless of data.
- **[KNOWN]** — commonly reported about the standard Kaggle-style CSV. *Verify against your file.*
- **[EXPECTED]** — a hypothesis about behaviour. Not a result. Never present it as one.
- **[RESULT]** — only numbers produced by running `g15_adaboost_social_network_ads.py` on your CSV.
- **[INTERPRETATION]** — what you conclude from [RESULT].

---

## 1. Project understanding

**What the ISE document actually says** (I read the uploaded file):
- ISE 2, topic *Ensemble Learning*, 10 marks, group activity.
- Each group gets a **problem, ML models and an ensemble technique** — all fixed; topic changes need prior approval.
- Required content: problem + real-world application; dataset, input features, target; base models; chosen ensemble technique; its working; implementation methodology; individual-model performance; ensemble performance; comparison and analysis; conclusion and references.
- Stated focus: *how multiple models are combined and whether the ensemble improves performance.*
- Rubric headings (no mark split given): understanding of the topic, quality of the presentation, report quality, timely submission.

**What the document does NOT say** (ask your faculty, don't assume): slide limit, time per group, report length/format, whether a notebook must be submitted, the deadline, the presentation date. The schedule is on a SharePoint sheet I can't open.

**What your group was assigned** (confirmed by your screenshot, row G15): Social Network Ads · Customer Purchase Prediction · Decision Tree-based learners · Boosting – AdaBoost.

**Central question:** *Does combining weak tree learners with AdaBoost improve purchase prediction compared with individual tree-based models?* The answer must come from your experiment. "No clear improvement" is a legitimate, gradable answer if you explain it.

**One important wording issue.** Random Forest and Extra Trees are *themselves* ensembles of trees. If you call them "individual models" an examiner will pounce. See section 6 for how to handle this honestly.

---

## 2. Recommended technical approach

| Decision | Recommendation | Why |
|---|---|---|
| Features | `Age`, `EstimatedSalary` only | These are the standard features; they make 2-D decision-boundary plots possible, which is the best visual for explaining AdaBoost. `User ID` is an identifier (no predictive meaning); `Gender` is optional (see note below). |
| Split | 80:20, `stratify=y`, `random_state=42` | Keeps class ratio equal in train and test; reproducible. |
| Scaling | **Not needed for trees.** Code proves this with a check, trains on raw features. | Tree splits are one-feature thresholds; monotonic rescaling doesn't change them. Scaling would only matter if you add a distance/gradient-based model, which is outside your topic. |
| Individual models | Decision Stump (depth 1), Shallow Tree (depth 3), Full Decision Tree | Real *single* trees spanning underfit → balanced → overfit. This gives AdaBoost fair opponents. |
| Contrast model | Random Forest, labelled "bagging reference" | Needed to explain bagging vs boosting. Clearly *not* an "individual" model. Extra Trees dropped to avoid scope creep. |
| Ensemble | AdaBoost with stumps (main) and depth-2 trees (variant) | Stumps are the textbook weak learner. |
| Sanity baseline | Majority-class `DummyClassifier` | Shows how much 64%-ish accuracy "for free" looks like. Not an algorithm you present as a model. |
| Evaluation | Hold-out test **plus** repeated stratified 5-fold CV (25 folds) | **Critical:** an 80-sample test set means one prediction = 1.25 percentage points. Differences between models of 1–3 points may be noise. CV gives mean ± std so you can say honestly whether AdaBoost's gain is real. |
| Tuning | Optional `GridSearchCV` on the **training** data only | Never choose hyperparameters by looking at the test set. |

*Gender note:* if your CSV has it, you may add it (one-hot or 0/1) as an extra experiment, but the main story and the boundary plots use Age and Salary. State this choice on the slide.

---

## 3. Dataset explanation

### 3.1 What it is
**[THEORY/KNOWN]** The Social Network Ads dataset is a small, educational dataset in which each row is one user who was shown an advertisement on a social network, with a label saying whether they went on to purchase the advertised product.

### 3.2 Structure — [KNOWN], verify
| Column | Meaning | Type | Use |
|---|---|---|---|
| `User ID` | Unique identifier | integer | **Drop** (identifier, no signal) |
| `Gender` | Male / Female | categorical | Not used in main experiment |
| `Age` | Age in years | integer | **Feature** |
| `EstimatedSalary` | Estimated annual salary | integer | **Feature** |
| `Purchased` | 0 / 1 | integer | **Target** |

Some uploads on the internet have only 3 columns, or different capitalisation (`Estimated Salary`). The script stops with a clear message if names don't match.

### 3.3 Row, features, target
- **A row** = one customer/user.
- **X = [Age, EstimatedSalary]**, **y = Purchased ∈ {0,1}**.
- **Mapping:** commonly 0 = did not purchase, 1 = purchased. *State that this is the usual encoding; confirm it from your file's source page.*
- **Classification, binary** — the output is a category, not a number.

### 3.4 Why ML suits it, and why it suits ensemble learning
- Purchase depends on a *combination* of features (older **and** higher-income users tend to buy more is the usual pattern — [EXPECTED], check with your scatter plot). Hand-writing rules is awkward; learning them from labelled history is the textbook use of supervised learning.
- The classes overlap in the Age–Salary plane, so one simple rule makes mistakes. That is precisely the setting where combining many simple rules can help.
- Only two features → you can *draw* what every model learned.

### 3.5 Facts you must NOT state until verified
| Item | Commonly reported [KNOWN] | How to get it from YOUR file |
|---|---|---|
| Samples | 400 | `df.shape[0]` |
| Columns | 5 | `df.shape[1]` |
| Dtypes | integers except Gender | `df.dtypes` |
| Class split | roughly 257 zeros / 143 ones (≈36% purchased) | `df['Purchased'].value_counts()` |
| Missing values | none | `df.isnull().sum()` |
| Duplicates | check | `df.duplicated(subset=['Age','EstimatedSalary','Purchased']).sum()` (User ID is unique, so full-row duplicates are meaningless) |
| Age range | roughly 18–60 | `df['Age'].describe()` |
| Salary range | roughly 15,000–150,000 | `df['EstimatedSalary'].describe()` |
| Outliers | usually none/few | IQR rule in the script; also eyeball the histogram |

The script's Section 1 prints all of these. **Copy the printed values into your slide, not the table above.**

*Provenance:* cite the exact web page you downloaded the CSV from (the dataset is widely mirrored on Kaggle; I won't guess a specific author/URL). Also say honestly that it is an educational dataset of unclear origin, so it should not be treated as evidence about real customers.

---

## 4. Problem statement

1. **One line:** *Predict whether a social-network user will purchase an advertised product from their age and estimated salary, and test whether AdaBoost over tree learners predicts better than individual trees.*

2. **Academic (2–3 sentences):** Targeted advertising requires estimating which users are likely to respond to an advertisement. We model this as a binary classification task on the Social Network Ads dataset, learning a mapping from Age and EstimatedSalary to Purchased ∈ {0,1}. Because single decision trees are either too simple (underfit) or too specific (overfit), we investigate whether AdaBoost, which sequentially combines weak tree learners, improves predictive performance over individual tree-based models.

3. **Presentation version:**
   - **Predict:** will this user buy? (0/1)
   - **Using:** Age, Estimated Salary
   - **Why:** show ads to the right people, cut wasted spend
   - **Question:** does AdaBoost beat single trees?

4. **Report version:** *"This work addresses customer purchase prediction in social-network advertising. Given a user's age and estimated salary, the objective is to classify whether the user purchased the advertised product. Tree-based classifiers are appropriate because they handle non-linear, threshold-style decision rules, need little preprocessing and are interpretable. However, a single tree has a bias–variance trade-off that depends strongly on its depth. We therefore compare individual decision trees of varying depth with AdaBoost, an ensemble that trains weak trees sequentially and combines them by weighted vote, using accuracy, precision, recall, F1-score and ROC-AUC on a held-out test set and under repeated stratified cross-validation."*

---

## 5. Real-world application: Customer Purchase Prediction

**Scenario.** A company advertises a product on a social platform. For every impression it pays. Historical records show age, estimated income and whether the user eventually bought. The company trains a classifier and uses it to decide whom to target.

**Uses**
| Use | How the model helps |
|---|---|
| Targeted advertising | Show the ad mainly to users predicted 1 |
| Customer segmentation | Leaf regions of a tree = readable segments, e.g. "Age > X and Salary > Y" |
| Campaign optimisation | Compare predicted-buyer share per campaign |
| Personalisation | Different creative for different segments |
| Lower ad cost | Fewer impressions on predicted non-buyers |
| Higher conversion rate | Same budget → more purchases |
| Finding interested customers | Rank users by predicted purchase probability |

**Careful wording (say this on the slide or aloud):** the model *estimates the class or likelihood of purchase from patterns in historical data*. It cannot predict an individual's behaviour with certainty. Two features cannot capture intent, price sensitivity, brand awareness or timing.

---

## 6. Base models and why they're chosen

### 6.1 The recommended set
| Model | Role in your story | Type |
|---|---|---|
| **Decision Stump** (`max_depth=1`) | The "weak learner". One question. | Individual tree |
| **Shallow Decision Tree** (`max_depth=3`) | A moderately complex, pruned-style tree | Individual tree |
| **Full Decision Tree** (no depth limit) | Maximum flexibility — likely to memorise training data | Individual tree |
| **Random Forest** (100 trees) | Contrast: parallel trees, bagging | Tree *ensemble* (reference) |
| **AdaBoost** (stumps, 50) | The assigned ensemble | Boosting ensemble |
| **AdaBoost** (depth-2 trees, 50) | Variant — shows weak-learner strength matters | Boosting ensemble |

**Why not call Random Forest/Extra Trees "individual models":** a forest averages many trees. The honest framing is *"individual trees"* (stump, shallow, full) vs *"tree ensembles"* (Random Forest = bagging, AdaBoost = boosting). This framing makes your **bagging-vs-boosting** explanation stronger, not weaker. It doesn't change your assigned topic (every model is tree-based).

### 6.2 Per-model explanation

**Decision Stump (depth 1)**
- *What:* a tree with a single split, e.g. "Age ≤ 36.5?".
- *Tree-based because* it is a decision tree, just one level.
- *Hyperparameters:* `max_depth=1` (fixed), `criterion` (gini default).
- *Pros:* extremely fast, interpretable, can't overfit much.
- *Cons:* only one feature, one threshold → can't represent "old AND high-income".
- *[EXPECTED]* underfits; likely the weakest.
- *Why included:* it is the standard AdaBoost weak learner, so it is the fairest "before" picture.

**Shallow Tree (depth 3)**
- *What:* at most 3 questions along any path (≤ 8 leaves).
- *Hyperparameters:* `max_depth`, `min_samples_leaf`, `ccp_alpha` (pruning).
- *Pros:* can combine Age and Salary; still readable on a slide.
- *Cons:* hard cut-offs; sensitive to training sample.
- *[EXPECTED]* a reasonable single-tree benchmark.
- *Why:* represents "a well-controlled single tree" — the opponent AdaBoost must beat to prove its worth.

**Full Decision Tree**
- *What:* grows until leaves are pure.
- *Pros:* zero bias on training data; learns fine detail.
- *Cons:* high variance; memorises noise.
- *[EXPECTED]* training accuracy at or near 100%, lower test accuracy than training → classic overfitting. Check `Train Acc` vs `Test Acc` in your table.
- *Why:* demonstrates the overfitting that ensembles try to tame.

**Random Forest (reference)**
- *What:* many full trees, each on a bootstrap sample and random feature subsets, majority vote.
- *Pros:* reduces variance; robust.
- *Cons:* less interpretable; with only 2 features the "random feature subset" has little room to diversify.
- *Why:* to contrast **parallel/independent** combination with AdaBoost's **sequential/adaptive** combination.

---

## 7. Ensemble learning explanation

**What:** combining several models so the group predicts better than a typical member.

**Why it can work** — three intuitions:
1. *Statistical:* many hypotheses fit the training data; averaging reduces the risk of picking a poor one.
2. *Variance reduction:* individual errors partly cancel (the idea behind bagging).
3. *Bias reduction:* many simple rules added together can represent a more complex boundary than any one rule (the idea behind boosting).

**Condition for success:** members must be *better than random* **and** *make different mistakes* (diversity). Ten copies of the same model give nothing.

| Approach | Training | Combination | Mainly reduces | Example |
|---|---|---|---|---|
| **Bagging** | Parallel, each model on a bootstrap sample | Majority vote / average | Variance | Random Forest |
| **Boosting** | Sequential; each model corrects predecessors | Weighted vote | Bias (and sometimes variance) | **AdaBoost** |
| **Stacking** | Different models; a meta-model learns how to combine | Meta-learner | Both, depends | Stacked classifier |

**Comparison table (use on the slide)**
| | Decision Tree | Random Forest | AdaBoost |
|---|---|---|---|
| # models | 1 | many (100) | many (50) |
| Tree depth | any | deep | shallow (usually 1–3) |
| Training | — | independent / parallel | **sequential** |
| Data for each tree | all | bootstrap sample + random features | **all data, re-weighted** |
| Vote | — | equal | **weighted by α** |
| Focus | — | diversity by randomness | **focus on previously misclassified samples** |
| Typical weakness | overfits | slower, less interpretable | **sensitive to noisy labels/outliers** |

---

## 8. AdaBoost in detail

**Adaptive Boosting.** Freund & Schapire, 1997.

**Idea in one sentence:** train a series of weak classifiers, each one concentrating on the examples earlier ones got wrong, then combine them in a vote where better classifiers get a bigger say.

**Vocabulary**
- **Weak learner:** only needs to be better than random guessing (error < 0.5). Here: a stump.
- **Sample weight wᵢ:** how much training sample *i* matters to the *next* learner. Starts equal.
- **Learner weight αₜ:** how much learner *t* matters in the final vote.

**Steps (customer purchase version)**
1. **Initialise** equal weights: wᵢ = 1/N for the N training customers.
2. **Train** weak learner h₁ on the weighted data (e.g. "Age ≤ 36.5?").
3. **Find mistakes:** which customers does h₁ misclassify?
4. **Weighted error** err₁ = total weight of those customers.
5. **Learner weight** α₁ from err₁ — low error ⇒ large α.
6. **Update sample weights:** raise weights of misclassified customers, lower the rest, renormalise to sum 1.
7. **Train h₂** on re-weighted data. It will prefer splits that fix h₁'s mistakes (e.g. a Salary split).
8. **Repeat** for T rounds (`n_estimators`).
9. **Final prediction:** weighted vote of all learners. Positive total ⇒ "Purchased".

### Toy numerical example (hypothetical — NOT your dataset)
10 training customers, labels ±1, so initial wᵢ = 0.10.

*Round 1.* Stump h₁ misclassifies 3 customers → err₁ = 3×0.10 = **0.30**.
α₁ = ½ ln(0.7/0.3) = ½ ln 2.333 = **0.424**.

Update:
- misclassified: 0.10 × e^(+0.424) = 0.10 × 1.528 = 0.1528
- correct: 0.10 × e^(−0.424) = 0.10 × 0.655 = 0.0655
- sum = 3×0.1528 + 7×0.0655 = 0.9165 (= normalising constant Z)
- normalised: misclassified 0.1528/0.9165 = **0.1667 (=1/6)**, correct 0.0655/0.9165 = **0.0714 (=1/14)**

Check: 3/6 + 7/14 = 0.5 + 0.5 = 1. **After the update the 3 hard customers hold exactly half of all weight** — this is why the next stump is forced to care about them (true for binary AdaBoost whenever err < 0.5).

*Round 2.* Suppose h₂ misclassifies 2 customers that h₁ got right: err₂ = 2 × 1/14 = **0.143**, α₂ = ½ ln(0.857/0.143) = ½ ln 6 = **0.896**.

*Round 3.* Suppose err₃ = 0.20 → α₃ = ½ ln 4 = **0.693**.

*Final vote for a new customer x* with h₁ = +1, h₂ = −1, h₃ = +1:
Σ αₜhₜ = 0.424 − 0.896 + 0.693 = **+0.221 > 0 → predict Purchased.**
The single most trusted learner (h₂) said "no", but two other learners together outweighed it. That is the weighted vote.

*(Numbers verified by calculation.)*

---

## 9. AdaBoost mathematics

Labels are coded **yᵢ ∈ {−1, +1}** in the maths (your CSV's 0 becomes −1).

**1. Weighted error of learner t**
`errₜ = Σᵢ wᵢ · I( yᵢ ≠ hₜ(xᵢ) )`  (weights sum to 1)
- wᵢ: current weight of sample i · I(·): 1 if the statement is true else 0 · hₜ(xᵢ): learner's prediction.
- Meaning: the fraction of *importance* the learner gets wrong — not the fraction of samples.

**2. Learner weight**
`αₜ = ½ · ln( (1 − errₜ) / errₜ )`
- Meaning: the "trust" given to learner t.

| errₜ | (1−err)/err | αₜ | Interpretation |
|---|---|---|---|
| near 0 (e.g. 0.05) | 19 | large positive (≈1.47) | very trusted; err = 0 gives infinite α, so implementations clip/stop |
| 0.30 | 2.33 | 0.42 | useful |
| **0.50** | 1 | **0** | no better than a coin flip → no influence |
| > 0.50 | < 1 | **negative** | worse than random; the formula would flip its vote. **scikit-learn instead stops/discards such a learner.** |

**3. Sample-weight update**
`wᵢ ← wᵢ · exp( −αₜ · yᵢ · hₜ(xᵢ) )`, then divide by the sum so weights add to 1.
- If correct: yᵢhₜ(xᵢ) = +1 → factor e^(−α) < 1 → weight **shrinks**.
- If wrong: yᵢhₜ(xᵢ) = −1 → factor e^(+α) > 1 → weight **grows**.
- The better the learner (bigger α), the more sharply it shifts attention.

**4. Final classifier**
`H(x) = sign( Σₜ αₜ · hₜ(x) )`  — t = 1…T.
- Positive ⇒ class +1 (Purchased); negative ⇒ −1.

**Assumptions / conditions:** binary labels in {−1,+1}; weights non-negative and normalised; every weak learner has errₜ < 0.5 on the weighted data; the weak learner can use sample weights (scikit-learn's trees do, via `sample_weight`).

**Variant warning for viva — scikit-learn's α differs by a factor of 2.** For two classes scikit-learn's SAMME algorithm uses `α = learning_rate × ln((1−err)/err)` — no ½. Every α is doubled, the sign of the weighted sum is unchanged, and the sample-weight update ends up identical after normalisation. The script prints both so you can show it. (You can see the difference between sklearn's α and the from-scratch α in the script's output.)

**Why AdaBoost works (one level deeper, optional):** it can be shown to minimise the exponential loss Σ exp(−yᵢF(xᵢ)) by greedily adding one weak learner at a time (Friedman, Hastie & Tibshirani, 2000). Also, exponential loss penalises badly misclassified points heavily — which explains the noise sensitivity.

---

## 10. Implementation methodology

```
 Social_Network_Ads.csv
         │
 1 Load → inspect (shape, dtypes, missing, duplicates, class balance, outliers)
         │
 2 EDA plots (class counts, Age/Salary histograms, scatter)
         │
 3 X = [Age, EstimatedSalary], y = Purchased
         │
 4 Stratified 80:20 split, random_state = 42
         │
 5 (Scaling check: show trees don't need it; fit scaler on TRAIN only)
         │
 6 Train: Stump │ Depth-3 tree │ Full tree │ (Random Forest ref.) │ (Majority baseline)
         │
 7 Train AdaBoost: stumps ×50, depth-2 trees ×50
         │
 8 Predict test set → Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
         │
 9 Repeated stratified 5-fold CV (25 folds) → mean ± std per model
         │
10 Compare individual vs ensemble → analyse → conclude
```

**Preprocessing questions you'll be asked**
- *Why scale if trees don't need it?* They don't. Scaling matters for models that use distances or gradient descent (k-NN, SVM, logistic regression, neural nets). The only reason to scale here is *consistency if you later add such a model*, or for readability of plots. Our script proves invariance rather than assuming it. Fit the scaler on **train only**; fitting on all data leaks test information.
- *Why a fixed `random_state`?* So split, tree tie-breaking, bootstrap samples and AdaBoost behaviour are the same every run → reproducible tables, and anyone can verify them.
- *Why stratify?* With ≈36% positives, an unlucky random split could put noticeably different class ratios in train and test.

**AdaBoost API notes (for current scikit-learn)**
- Use `AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0, random_state=42)`.
- The old argument name `base_estimator` was deprecated in 1.2 and removed in 1.4; `estimator` replaces it.
- The `algorithm` argument ('SAMME.R' vs 'SAMME') is deprecated from 1.6; the library now always uses SAMME. The script omits it on new versions and sets `'SAMME'` on versions older than 1.6 (so behaviour matches).
- Tested here on scikit-learn 1.8.0.

**Hyperparameters — don't choose blindly**
- `estimator` (weak learner): shallower = weaker and more stable; deeper = stronger but more prone to overfit and gives each round more influence.
- `n_estimators`: more rounds → fits training data better; stops early if a learner is perfect or worse than random.
- `learning_rate`: shrinks each learner's contribution. Lower values usually need more estimators; there is a trade-off between the two.
- The script includes (a) an illustration of how test accuracy changes with rounds and learning rate (for **understanding only**) and (b) `GridSearchCV` on the training set for real tuning.

---

## 11. Complete Python code

**File:** `g15_adaboost_social_network_ads.py` (delivered with this guide).

**To run:** put `Social_Network_Ads.csv` beside the script → `pip install pandas numpy matplotlib seaborn scikit-learn` → `python g15_adaboost_social_network_ads.py`.

**What it does (sections):** 1 load/inspect · 2 EDA plots · 3 split · 4 scaling sanity check · 5 models · 6 hold-out results · 7 confusion matrices, bar charts, ROC · 8 decision boundaries, AdaBoost accuracy-vs-rounds, errors and α per round, learning-rate effect, a depth-3 tree plot · 9 repeated CV · 10 optional tuning (training data only) · 11 from-scratch AdaBoost teaching demo.

**Outputs:** 13 PNG files in `./figures/`, two CSVs in `./results/`.

**Honesty about testing:** I checked that the script runs end-to-end without errors using a synthetic stand-in file I generated, because I don't have your real CSV. Those synthetic numbers are meaningless and I have not reported or used them anywhere. All your numbers will come from your own run.

**Viva-friendly explanation of the code:** each numbered section has a comment header; every model is one line in the `models` dictionary; section 11 re-implements AdaBoost in ~20 lines so you can explain every formula in code.

---

## 12. Evaluation metrics

Confusion-matrix terms for this problem (positive class = Purchased):
- **TP:** predicted purchase, actually purchased.
- **TN:** predicted no purchase, actually no purchase.
- **FP:** predicted purchase, actually no purchase.
- **FN:** predicted no purchase, actually purchased.

| Metric | Formula | Meaning | Why it matters here |
|---|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | share of all predictions correct | Simple overall view; misleading under imbalance (a model saying "nobody buys" scores roughly the majority share) |
| Precision | TP/(TP+FP) | of users we said would buy, how many did | Controls wasted ad spend |
| Recall | TP/(TP+FN) | of real buyers, how many we found | Controls missed customers |
| F1 | 2·P·R/(P+R) | harmonic mean of P and R | One number balancing both; low if either is low |
| Confusion matrix | 2×2 counts | exactly what kind of errors | Shows *which* mistakes AdaBoost changes |
| ROC-AUC | area under TPR-vs-FPR curve over all thresholds | probability a random buyer is scored above a random non-buyer | Threshold-free ranking quality — useful if you rank customers by probability. Note single trees give coarse probabilities (a full tree gives only 0/1), so its AUC is not directly comparable in smoothness |

**Why not accuracy alone:** it hides the error mix. Two models can have equal accuracy while one wastes budget (FP) and the other loses customers (FN). The majority baseline shows that accuracy has a floor of ≈ the majority-class share just by guessing.

**Confusion-matrix interpretation**
```
                Predicted
               0      1
Actual   0    TN     FP
         1    FN     TP
```
**Business trade-off (neither error is universally worse):**
- If each impression is *expensive* and margin per sale is *small* → FP hurts more → favour **precision**.
- If margin per sale is *large* and impressions *cheap* → FN hurts more (lost sale) → favour **recall**.
- Ask: "what is the cost of showing one wasted ad vs. missing one buyer?" That ratio decides.

---

## 13. Results table template (fill from your run — do not invent)

**Table A — hold-out test set (n ≈ 80 if you have 400 rows)** → `results/holdout_results.csv`

| Model | Group | Train Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Majority baseline | Baseline | | | | | | |
| Decision Stump (depth 1) | Individual tree | | | | | | |
| Shallow Tree (depth 3) | Individual tree | | | | | | |
| Full Decision Tree | Individual tree | | | | | | |
| Random Forest (ref.) | Bagging reference | | | | | | |
| AdaBoost (stumps, 50) | Boosting | | | | | | |
| AdaBoost (depth-2, 50) | Boosting | | | | | | |

**Table B — repeated stratified 5-fold CV, mean ± std over 25 folds** → `results/cv_results.csv`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| (same rows as above) | | | | | |

**Table C — confusion-matrix counts** (printed by script): TN / FP / FN / TP per model.

**Table D (optional)** — tuned AdaBoost: best parameters, CV F1 on train, test metrics.

### How to interpret once filled in
1. **Look at Train Acc vs Test Acc.** A big gap = overfitting (expect it for the full tree [EXPECTED]).
2. **Compare AdaBoost vs the best individual tree**, not vs the weakest. The stump is a weak opponent; the fair comparison is against the best single tree.
3. **Translate differences into samples.** On 80 test samples, +1.25 points = *one* extra correct prediction. Don't call that "significant".
4. **Check the CV table.** If AdaBoost's mean is higher by less than about one standard deviation, say "comparable/slightly higher, not clearly better".
5. **Check precision and recall separately.** AdaBoost may raise one and lower the other — report the trade-off.
6. **Check Random Forest too.** If it ties AdaBoost, say so; it is a legitimate finding.
7. **Conclusion templates** (pick the one your data supports):
   - *Improvement:* "AdaBoost (stumps) achieved X vs Y for the best single tree on the test set and a CV mean of A ± s vs B ± s; the gain is [larger than/comparable to] fold-to-fold variation."
   - *No clear difference:* "AdaBoost performed comparably to a depth-3 tree (X vs Y; CV overlap). On a 2-feature, 400-sample dataset a single well-controlled tree is already close to the achievable accuracy, so ensembling adds little."
   - *Worse:* "AdaBoost did not improve …; possible reasons are label noise/overlap between classes, which AdaBoost's focus on hard samples can over-emphasise."

Ask me to analyse the output once you paste it (the tables or the two CSV files).

---

## 14. Graphs to generate

All are produced by the script. Put at most one or two per slide.

| # | File | Shows | Why useful | Slide | Takeaway for audience |
|---|---|---|---|---|---|
| 1 | `01_purchase_distribution.png` | counts of 0 vs 1 | exposes imbalance | 5 | "Classes aren't 50/50, so accuracy alone isn't enough" |
| 2 | `02_feature_distributions.png` | Age and Salary histograms by class | feature ranges, overlap | 5 | "Buyers skew older / higher salary" **only if the plot shows it** |
| 3 | `03_age_salary_scatter.png` | 2-D scatter by class | classes overlap → single rule can't be perfect | 4 or 5 | "Why a simple model errs; why combining may help" |
| 4 | `12_decision_tree_depth3.png` | actual tree learned | makes trees concrete | 7 | "This is how a purchase prediction is made" |
| 5 | `08_decision_boundaries.png` | regions learned by stump, full tree, RF, AdaBoost | the best visual of ensemble vs single | 14/15 | "Stump = one cut, full tree = jagged, AdaBoost = layered cuts" |
| 6 | `10_adaboost_errors_and_alphas.png` | per-round weighted error and α | shows weighting working | 9–10 | "Later learners face harder problems (error rises); α reflects trust" |
| 7 | `09_adaboost_accuracy_vs_learners.png` | train/test accuracy as learners are added | n_estimators effect, overfitting check | 14 | "Does adding learners help, and when does it plateau?" |
| 8 | `04_confusion_matrices.png` | per-model TN/FP/FN/TP | error types | 13–15 | "What kind of mistakes changed" |
| 9 | `05_accuracy_comparison.png` | bar chart of test accuracy | headline comparison | 15 | answer to central question |
| 10 | `06_precision_recall_f1.png` | grouped bars | trade-offs | 15 | "Precision/recall balance" |
| 11 | `07_roc_curves.png` | ROC for all models | ranking quality | 15 | "Which model ranks buyers best" |
| 12 | `13_cv_accuracy_boxplot.png` | accuracy across 25 folds | tests whether differences are real | 15 | "Is the gain larger than random variation?" |
| 13 | `11_learning_rate_effect.png` | effect of learning_rate | hyperparameter intuition | backup | "Why parameters aren't arbitrary" |

**Slide hygiene:** one main chart per slide; ≥ 18 pt axis text (the script uses `context="talk"`); no more than 7 models in any chart; use consistent colours (see design guide).
