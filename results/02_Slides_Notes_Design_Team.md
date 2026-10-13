# ISE 2 — Group G15 — Slides, Speaker Notes, Design, Team Split (Part 2 of 4)

Deliverables 15, 16, 20, 21. Slide text is deliberately short; the explanation lives in the speaker notes.

**Structure: 16 content slides + 1 references slide.** The ISE document sets no slide limit or time; check the time slot with your faculty. If you must cut, merge slides 2+3, and 12 into 11.

**Placeholders** like `[N]` or `[from Table A]` must be filled from *your* run. Never leave a placeholder or invented number on a final slide.

**Official requirement → slide map**
| ISE requirement | Slide(s) |
|---|---|
| Problem statement & application | 2, 3 |
| Dataset, features, target | 4, 5 |
| Base models | 6, 7 |
| Ensemble technique | 8, 9 |
| Working of the ensemble | 9, 10 |
| Implementation methodology | 5, 11 |
| Individual model performance | 13 |
| Ensemble performance | 14 |
| Comparison & analysis | 15 |
| Conclusion & references | 16, 17 |

---

## Slide 1 — Title

**Slide content**
- **Does AdaBoost Improve Customer Purchase Prediction?**
- Boosting weak decision trees on the Social Network Ads dataset
- ISE 2 · Ensemble Learning · Group G15
- Team members (names, roll numbers) · Course · Faculty name · Date

**Speaker notes (≈30 s)**
"Good [morning]. We're Group G15. Our topic is ensemble learning. We were assigned the Social Network Ads dataset, customer purchase prediction as the application, decision-tree-based learners as base models, and AdaBoost as the ensemble technique. We'll show how AdaBoost combines trees and, using our own experiments, whether it actually beats individual trees."

**Visual:** clean title; right side a faint stylised image of a single stump → several stumps → one combined boundary.
**Emphasise:** *AdaBoost, weak learners, "does it actually help?"*

---

## Slide 2 — Problem Statement

**Slide content**
- **Predict:** will a social-network user purchase the advertised product? (0 / 1)
- **From:** Age, Estimated Salary
- **Why:** show ads to likely buyers, reduce wasted spend
- **Why trees:** threshold-style rules, interpretable
- **Central question:** *Does AdaBoost over weak trees beat individual trees?*

**Speaker notes (≈60 s)**
"Our task is binary classification. Given a user's age and estimated salary, we predict whether they purchased. Advertising costs money per impression, so knowing who's likely to buy is valuable. Tree models are natural here because decisions look like 'age above X and salary above Y'. But a single tree is either too simple or memorises training data. So our central question is whether AdaBoost, which combines many weak trees, improves on single trees. The answer comes from our experiment, not assumption."

**Visual:** simple flow: *[Age, Salary] → Model → Purchased? 0/1*, with the central question in a highlighted banner.
**Emphasise:** *binary classification, central question.*

---

## Slide 3 — Real-World Application

**Slide content**
- Scenario: company runs ads on a social platform
- Model estimates **likelihood of purchase** from historical patterns
- Uses: targeted ads · segmentation · campaign optimisation · personalisation · lower ad cost · higher conversion
- ⚠ Estimates, not certainty — two features can't capture intent

**Speaker notes (≈60 s)**
"Imagine a company paying for every ad impression. If it can rank users by purchase likelihood, it can target ads, build customer segments, personalise creative and cut waste. Our model doesn't know what a person will do; it learns patterns from past data. Real systems would use many more features such as browsing behaviour, price sensitivity and timing."

**Visual:** 6 icons in a 3×2 grid (target, people groups, megaphone, user card, coin, upward arrow).
**Emphasise:** *likelihood, targeted advertising.*

---

## Slide 4 — Dataset, Features and Target

**Slide content** (use the numbers from your own file)
| Column | Role |
|---|---|
| Age | Input feature |
| EstimatedSalary | Input feature |
| Purchased | Target (0 = no, 1 = yes) |
| User ID, Gender | Not used in main experiment |
- Rows: `[N]` · Columns: `[N]` · Missing values: `[N]` · Duplicates: `[N]`
- Class split: `[n0] / [n1]` (`[x]`% purchased)
- Age range `[min–max]` · Salary range `[min–max]`
- Source: `[exact web page / Kaggle link you used]`

**Speaker notes (≈75 s)**
"Each row is one user. X is Age and Estimated Salary; y is Purchased. We exclude User ID because it's just an identifier, and we keep the main experiment to two features so we can visualise every model's decision regions. The class split is not 50/50, which is why we'll report precision, recall and F1 and not only accuracy. This is an educational dataset, so conclusions apply to the dataset, not to real customers in general."

**Visual:** left: table; right: `03_age_salary_scatter.png`.
**Emphasise:** *target = Purchased, class imbalance, what a row represents.*

---

## Slide 5 — Exploration & Preprocessing

**Slide content**
- Pipeline: Load → Inspect → Missing/duplicates → Class balance → X,y → **Stratified 80:20 split (random_state = 42)** → Models
- **Scaling:** trees are scale-invariant — verified: `[n]` prediction differences raw vs scaled
- Train `[n]` rows · Test `[n]` rows
- Charts: purchase distribution + Age/Salary histograms

**Speaker notes (≈75 s)**
"We checked missing values, duplicates and class balance, then split 80:20 with stratification so train and test keep the same class ratio, and a fixed random state so anyone can reproduce our numbers. We also tested feature scaling. Trees ask one-feature threshold questions, so rescaling shouldn't change their predictions, and our check confirmed [state your actual result]. Scaling matters for distance- or gradient-based models, which are outside our topic. One more thing: our test set has only [n] samples, so one prediction is worth [100/n] percentage points. That's why we also use repeated cross-validation."

**Visual:** pipeline arrow strip on top; two small charts below (`01`, `02`).
**Emphasise:** *stratified, random_state, scale-invariance of trees, small test set.*

---

## Slide 6 — Individual Tree-Based Models

**Slide content**
| Model | Idea | Expected role |
|---|---|---|
| Decision Stump (depth 1) | one question | weak learner |
| Shallow Tree (depth 3) | a few questions | balanced single tree |
| Full Decision Tree | until pure | overfitting risk |
| *Random Forest (reference)* | many trees, bagging | contrast with boosting |
- Why included: span underfit → balanced → overfit

**Speaker notes (≈75 s)**
"We use three individual trees of different depth. The stump is the classic weak learner. The depth-3 tree is a controlled single tree and our main opponent. The full tree shows what happens when a tree is allowed to memorise. We also include Random Forest, but we're clear that it is itself an ensemble. We use it only to contrast bagging, where trees are trained independently, with boosting, where they are trained one after another. Our expectations are hypotheses, and the results slides show what actually happened."

**Visual:** three small boundary sketches (one cut → few cuts → jagged); Random Forest box in grey with dashed outline labelled "ensemble (reference)".
**Emphasise:** *weak learner, overfitting, Random Forest is an ensemble.*

---

## Slide 7 — Decision Tree

**Slide content**
- Root → internal nodes → branches → leaves
- Split quality: **Gini** = 1 − Σpₖ² · **Entropy** = −Σpₖ log₂pₖ · **Information gain** = parent − weighted children
- Controls: `max_depth`, pruning
- Diagram: `Age ≤ X?` → Yes/No → `Salary ≤ Y?` → Predict 0/1  *(use your real depth-3 tree image `12_...png`)*
- Depth 1 = **stump** = weak learner

**Speaker notes (≈90 s)**
"A tree asks yes/no questions about features. The root asks the most useful question first. 'Useful' is measured by impurity: Gini or entropy. A good split makes child nodes purer; the drop in impurity is information gain. A leaf gives the prediction by majority class. If we let the tree grow without limit it fits noise: that's overfitting. We limit depth or prune. With depth 1 we get a stump: one threshold on one feature, which is just better than guessing. That's what AdaBoost uses as a weak learner." *(Optional hypothetical example: a node with 6 'no' and 4 'yes' has Gini 0.48; a split giving a pure left child and a 1-vs-4 right child lowers weighted Gini to 0.16.)*

**Visual:** your real tree plot.
**Emphasise:** *Gini, information gain, overfitting, stump.*

---

## Slide 8 — Ensemble Learning: Bagging vs Boosting

**Slide content**
- Ensemble = combine models → better than a typical single member
- Needs: members **better than random** + **diverse errors**
| | Bagging (RF) | **Boosting (AdaBoost)** |
|---|---|---|
| Training | parallel | **sequential** |
| Data | bootstrap samples | **re-weighted full data** |
| Vote | equal | **weighted** |
| Mainly reduces | variance | **bias** |
- Stacking: a meta-model combines base models (not used here)

**Speaker notes (≈75 s)**
"An ensemble combines several models. It helps only if members are individually better than random and make different mistakes: ten copies of the same model add nothing. Bagging trains deep trees in parallel on random samples and averages them, reducing variance. Boosting trains simple trees in sequence, each correcting the previous ones, and combines them with a weighted vote, mainly reducing bias. Stacking uses a meta-model; we mention it for completeness. Our assigned technique is boosting, specifically AdaBoost."

**Visual:** two-lane diagram: top lane parallel arrows to a vote box; bottom lane chain of boxes with growing/shrinking dots.
**Emphasise:** *diversity, sequential vs parallel, weighted vote.*

---

## Slide 9 — AdaBoost: The Idea

**Slide content**
- **Ada**ptive **Boost**ing (Freund & Schapire, 1997)
- Weak learners trained **sequentially**
- Hard (misclassified) samples get **more weight**
- Better learners get **bigger vote (α)**
- Flow diagram (see design section for layout)

**Speaker notes (≈90 s)**
"AdaBoost starts with every training customer equally important. It trains a weak tree, then looks at its mistakes. Customers it got wrong get more weight and those it got right get less. The next tree trains on this re-weighted problem, so it pays attention to the hard cases. Each tree also gets a score alpha based on how accurate it was. At the end, all trees vote, but trees with larger alpha count more. Note that AdaBoost is not itself a tree. It is a method of combining learners; we choose trees as the learners."

**Visual:** the vertical flow diagram below; **animate** reveal round by round (see Part 28 section).
**Emphasise:** *sample weights, α, weighted vote, AdaBoost is not a tree.*

---

## Slide 10 — AdaBoost: The Mathematics

**Slide content**
- `errₜ = Σ wᵢ · I(yᵢ ≠ hₜ(xᵢ))` — weighted error
- `αₜ = ½ ln((1 − errₜ) / errₜ)` — learner weight
- `wᵢ ← wᵢ · exp(−αₜ yᵢ hₜ(xᵢ))`, then normalise — weight update
- `H(x) = sign(Σ αₜ hₜ(x))` — final prediction
- err ≈ 0 → large α · **err = 0.5 → α = 0** · err > 0.5 → negative α (sklearn stops)
- **Toy example (hypothetical):** err = 0.30 → α = 0.42 → hard samples hold 1/2 of total weight

**Speaker notes (≈120 s)**
"Weighted error is the total weight of the samples the learner gets wrong. Alpha converts error to trust: a learner with error 0.05 gets a big alpha, a coin-flip learner gets zero, and a worse-than-random learner would get a negative alpha, which scikit-learn avoids by stopping. For the update, correct samples are multiplied by e to the minus alpha and wrong samples by e to the plus alpha, so mistakes grow and successes shrink; we renormalise to sum to one. In our toy example with 10 samples and 3 mistakes, alpha is 0.42 and after normalisation the 3 mistakes carry half of the total weight. The final prediction is the sign of the weighted sum. Note that scikit-learn uses alpha without the one-half factor; it doubles every alpha and doesn't change the prediction."

**Visual:** equations in boxes, each with a one-line English label; toy-example mini table on the right.
**Emphasise:** *weighted error, α, normalisation, sign of weighted sum.*

---

## Slide 11 — Implementation Methodology

**Slide content**
- Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
- Split: 80:20 stratified, `random_state = 42`
- Models: Stump, Depth-3, Full tree, (RF reference), **AdaBoost (stumps ×50)**, AdaBoost (depth-2 ×50)
- AdaBoost: `AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0)`
- Evaluation: hold-out test + **repeated stratified 5-fold CV (25 folds)**
- Tuning (optional): `GridSearchCV` on train data only

**Speaker notes (≈75 s)**
"Everything is in scikit-learn. We trained every model on the same training set and evaluated on the same test set. Because the test set is small, we also ran repeated stratified cross-validation and report mean and standard deviation. AdaBoost's main settings are the weak learner, number of estimators and learning rate. We did not guess: [state what you did, e.g., default 50/1.0 as the main run, plus a grid search on training data only]. We never tuned on the test set."

**Visual:** the horizontal pipeline from Part 10 of the guide; code snippet (3 lines) in monospace.
**Emphasise:** *same split for all models, CV, no tuning on the test set.*

---

## Slide 12 — Evaluation Metrics

**Slide content**
- Confusion matrix: TN FP / FN TP (positive = Purchased)
- Accuracy = (TP+TN)/all · Precision = TP/(TP+FP) · Recall = TP/(TP+FN) · F1 = 2PR/(P+R) · ROC-AUC = ranking quality
- **FP** = wasted ad · **FN** = missed buyer
- Accuracy alone can hide this

**Speaker notes (≈75 s)**
"A false positive means we predicted a purchase and the user didn't buy: a wasted ad. A false negative means we predicted no purchase but they would have bought: a missed customer. Precision measures how clean our predicted buyers are, recall how many real buyers we find, F1 balances them. Which error matters more depends on cost: if ads are expensive and margins thin, precision matters more; if margins are high, recall does. So we don't rely on accuracy alone. ROC-AUC measures how well the model ranks buyers above non-buyers across all thresholds."

**Visual:** 2×2 colour-coded confusion matrix with business meaning in each cell.
**Emphasise:** *FP vs FN business meaning, why not accuracy alone.*

---

## Slide 13 — Results: Individual Tree Models

**Slide content** (all numbers from Table A/B)
- Table: Stump, Depth-3, Full tree — Train Acc, Test Acc, Precision, Recall, F1, ROC-AUC `[fill]`
- Chart: `04_confusion_matrices.png` (individual trees only) or `05` bars
- One-line observation `[write only what your table shows, e.g. "Full tree: train [x] vs test [y] → overfitting"]`

**Speaker notes (≈75 s)**
"Here are the individual trees. [Read your numbers.] Compare training with test accuracy for each: a large gap means overfitting. The stump [state result] and the full tree [state result]. We also report the majority-class baseline of [x]% so you can see how much better the models are than guessing."

**Visual:** compact table + one chart. Highlight the best single tree in the table (it is the benchmark for the next slides).
**Emphasise:** *train vs test gap, best single tree = fair opponent.*

---

## Slide 14 — Results: AdaBoost

**Slide content**
- AdaBoost (stumps ×50): Acc `[ ]` · Prec `[ ]` · Rec `[ ]` · F1 `[ ]` · AUC `[ ]`
- AdaBoost (depth-2 ×50): `[ ]`
- Chart left: `09_adaboost_accuracy_vs_learners.png`
- Chart right: `08_decision_boundaries.png` (stump vs AdaBoost)
- Note: first-round vs later-round weighted error `[from 10_...png]`

**Speaker notes (≈90 s)**
"AdaBoost with 50 stumps scored [..]. This plot shows accuracy as we add learners one at a time: [describe the real curve: rise, plateau, any train–test divergence]. On the right, a single stump makes one straight cut, while AdaBoost's boundary is made of many cuts layered together. Notice in the errors-per-round plot that [describe]: later stumps typically face harder reweighted data, so their weighted errors tend to rise towards 0.5 — check whether your plot shows that."

**Visual:** two charts side by side; small results box.
**Emphasise:** *layered boundary, more learners ≠ always better.*

---

## Slide 15 — Comparison & Analysis

**Slide content**
- One table: all models, hold-out metrics + CV mean ± std (Table A + B condensed)
- Chart: accuracy bars or `13_cv_accuracy_boxplot.png`; ROC `07` on a backup slide
- **Findings** (3–4 bullets, written *after* you see results):
  - Best single tree: `[ ]` · AdaBoost: `[ ]` · Difference `[ ]` = `[k]` test samples
  - Precision/recall change: `[ ]`
  - CV says: `[improvement is / is not larger than fold variation]`
  - Random Forest vs AdaBoost: `[ ]`

**Speaker notes (≈120 s)**
"This is the core of our project. Against the best single tree, AdaBoost [improved / was comparable / did not improve] accuracy by [x] points, which is [k] test samples. On the 25 CV folds the means are [A ± s] and [B ± s], so [state whether the gap exceeds the spread]. Precision [..] and recall [..], so [state trade-off]. Random Forest [..]. Our conclusion for this dataset: [be exact, e.g. 'ensemble gives a small but consistent gain', or 'a depth-3 tree is already near the ceiling']. We don't claim AdaBoost always wins; it depends on the data."

**Visual:** big table with the best value in each column bolded; one chart.
**Emphasise:** *difference in samples, fold variation, trade-off.*

---

## Slide 16 — Conclusion, Limitations, Future Scope

**Slide content**
- **Answer to central question:** `[from your results]`
- **How AdaBoost works:** sequential weak trees + sample re-weighting + weighted vote
- **Limits:** small educational dataset (400 rows, 2 features) · test set small · noise-sensitive · results are split-dependent
- **Future:** cross-validated tuning · more features · gradient boosting/XGBoost comparison · explainability · deployment

**Speaker notes (≈60 s)**
"To conclude: [your answer]. AdaBoost improves by re-weighting hard samples and combining weak trees through a weighted vote. Our conclusions are limited by a small, simple dataset and a small test set, and AdaBoost can be sensitive to noisy labels. As future work, we would use more features, run cross-validated tuning, and compare with other boosting methods such as gradient boosting and XGBoost, plus explanation tools."

**Visual:** 3 columns: Answer / Limits / Future.
**Emphasise:** *the honest answer, limits.*

---

## Slide 17 — References

**Slide content** (use the exact list in Part 3 of this guide; at minimum:)
1. Freund & Schapire (1997), JCSS 55(1).
2. Pedregosa et al. (2011), JMLR 12.
3. scikit-learn User Guide — Ensembles; `AdaBoostClassifier` documentation.
4. Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2nd ed.).
5. Breiman (2001), Random Forests.
6. Dataset page `[URL, access date]`.

**Speaker notes:** "Thank you. We're happy to take questions." (leave this slide up during Q&A)

---

# Design Recommendations (Part 28)

**Palette**
| Use | Colour |
|---|---|
| Background | `#F7F9FC` (off-white) or deep navy `#1B2A41` for title/section slides |
| Primary text | `#1B2A41` |
| Individual trees | blue `#4C78A8` |
| AdaBoost / ensemble | orange `#E07B39` |
| Errors / misclassified | red `#C8453B` |
| Neutral / reference (Random Forest, baseline) | grey `#8A94A6` |
Rule: **colour = meaning, used the same on every slide and in every chart** (the script already uses blue / grey / orange).

**Fonts:** headings Montserrat SemiBold (or Calibri Bold if you need a safe default); body Inter or Open Sans; code/equations Consolas or JetBrains Mono. Body ≥ 20 pt, titles 32–36 pt, chart text ≥ 16 pt.

**Layout:** one idea per slide; max 5 bullets of ≤ 8 words; 16:9; thin accent bar at left of titles; slide number and "G15 · ISE 2" in footer.

**Diagrams:** rounded rectangles, one arrow style, left-to-right for pipelines and top-to-bottom for AdaBoost flow.

**Tables:** header row navy with white text, light zebra rows, best value per column in bold. No more than 7 rows.

**Graphs:** Matplotlib `talk` style as in the script; same model order and colours everywhere; titles state the takeaway.

**Icons:** a single line-icon set (Lucide, Feather or Material outline). No clip-art.

**Animations — only where they teach**
- **Slide 9 (AdaBoost flow):** reveal step by step: dataset with equal-size dots → Learner 1 appears with a boundary → misclassified dots **grow** (red) and correct ones shrink → Learner 2 → … → final weighted combination. 4–5 builds, *appear* or *fade* only.
- **Slide 10:** optionally reveal the toy-example numbers line by line.
- Everywhere else: no animation.

### AdaBoost flow diagram layout (recreate in PowerPoint/Canva/Figma)
```
┌──────────────────────────────┐
│  Training data: Age, Salary   │      (grey box)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Step 1  Equal sample weights  │      (all dots same size)
│        wᵢ = 1/N               │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Weak Learner 1 (stump)        │ ───► α₁ (tag, orange)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Find misclassified samples    │      (red outlines)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ↑ weight of mistakes           │
│ ↓ weight of correct, normalise │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Weak Learner 2                │ ───► α₂
└──────────────┬───────────────┘
               ▼
        (again: focus on
         difficult samples)
               ▼
┌──────────────────────────────┐
│ Weak Learner 3 … T            │ ───► α₃ … α_T
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Weighted vote:                │
│ H(x) = sign(Σ αₜ hₜ(x))       │      (orange box)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Purchase prediction: 0 / 1    │      (navy box)
└──────────────────────────────┘
```
Put 3 small scatter thumbnails to the right of "Learner 1/2/3" showing dot sizes changing; a curved arrow from "update weights" back up to "next learner" signals the loop. Use **blue** for learners, **orange** for α and the final vote, **red** for misclassified samples.

---

# Team Distribution (Part 29)

Balance rule: theory-heavy slides go to people who also have fewer slides.

| Member | Slides | Speaking time (approx.) |
|---|---|---|
| **Member 1** | 1, 2, 3, 4, 17 | Title, problem, application, dataset (≈ 4 min) |
| **Member 2** | 5, 6, 7, 12 | Preprocessing, base models, decision tree, metrics (≈ 5 min) |
| **Member 3** | 8, 9, 10 | Ensemble learning, AdaBoost idea, AdaBoost maths (≈ 5 min) |
| **Member 4** | 11, 13, 14, 15, 16 | Implementation, all results, analysis, conclusion (≈ 6 min) |

Everyone must also be able to answer questions about the *whole* project at a basic level; examiners often ask a different person than the one who presented.

### Member 1 — Introduction & Data
- **Must understand:** supervised vs unsupervised; classification; features vs target; why 0/1; class imbalance; why User ID is dropped; limitations of the dataset; what "estimates likelihood" means.
- **Likely questions:** What is the target variable? Why is this classification? Why is this useful for a company? What are the features? Is the dataset balanced? What are its limitations? (Viva file: Q9, Q10, Q31, Q32, Q38.)
- **Extra duty:** fill in the dataset facts from the actual CSV; write the problem statement for the report.

### Member 2 — Preprocessing, Trees, Metrics
- **Must understand:** train/test split and why; stratification; random_state; scaling and why trees don't need it; Gini, entropy, information gain; depth, overfitting, pruning; accuracy, precision, recall, F1, confusion matrix, FP vs FN.
- **Likely questions:** Why split? Why stratify? Does scaling affect decision trees? What is Gini impurity? How can a tree overfit? Why not use accuracy only? Define precision and recall. (Viva file: Q11–Q14, Q16–Q26, Q33, Q36, Q37, Q39, Q40; cross-question X6.)
- **Extra duty:** produce slide 7's real tree plot; write the metrics section of the report.

### Member 3 — Ensemble & AdaBoost Theory
- **Must understand:** ensemble idea; bagging vs boosting vs stacking; weak learner; stump; sample weights; α; the four formulas; what happens at err = 0, 0.5, >0.5; Random Forest vs AdaBoost; noise sensitivity; the factor-½ difference in scikit-learn.
- **Likely questions:** Viva file Q1–Q8, Q15, Q27–Q29, Q34, Q35 and cross-questions X1–X5, X7–X10.
- **Extra duty:** build the animated AdaBoost flow; rehearse the toy example by hand with a different set of numbers.

### Member 4 — Code, Results, Conclusion
- **Must understand:** every section of the script; how `AdaBoostClassifier` is configured; what each result file contains; train-vs-test gap; CV mean ± std; why differences of 1–2 samples aren't conclusive; how conclusions follow from the data.
- **Likely questions:** How do you know the ensemble improved? Which hyperparameters did you use and why? Did you tune on the test set? What would you change? Does AdaBoost always improve accuracy? (Viva file: Q2, Q27, Q30; cross-question X11.)
- **Extra duty:** run the code, save outputs, paste results to the group (and to me for analysis); own the report's results and discussion sections.

**Group tasks:** all four read the ISE document once; one person (suggest Member 1) owns the final submission and deadline; one rehearsal with a timer; all four check that every number on every slide matches `results/*.csv`.

HOW TO RUN
  1. Put Social_Network_Ads.csv in the same folder as this file
     (or change CSV_PATH below).
  2. pip install pandas numpy matplotlib seaborn scikit-learn
  3. python g15_adaboost_social_network_ads.py
  4. Figures are saved in ./figures and tables in ./results.
     Copy the printed tables / the CSV files into your slides and report.

Requires scikit-learn >= 1.2 (uses the `estimator=` argument of AdaBoost).
Nothing in this file contains made-up results: every number is computed
from YOUR CSV when you run it.