# ISE 2 — Group G15 — Report, Limitations, Future Scope, References, Checklist (Part 3 of 4)

Deliverables 17, 22, 23 plus Parts 25–27 of your brief.

The ISE document does not give a report length or format. Ask your faculty; a sensible default is 8–12 pages, A4, 11-pt font, numbered figures/tables, IEEE-style numbered references.

**Rule for every section below:** keep *theory*, *expected behaviour*, *actual results* and *interpretation* visibly separate. In Sections 14–16, only actual numbers from your run go in tables; interpretation sentences refer to those numbers.

---

# Report Structure and Content Guidance

## 1. Title
*"Customer Purchase Prediction on the Social Network Ads Dataset: Comparing Individual Decision Trees with AdaBoost"* — Group G15, names, roll numbers, course, faculty, date, ISE 2.

## 2. Abstract (150–200 words, write last)
Template (replace every bracket with your real outcome):
> Predicting which social-network users will purchase an advertised product supports targeted advertising. This work formulates the task as binary classification on the Social Network Ads dataset (`[N]` records; features Age and EstimatedSalary; target Purchased). Individual decision trees of depth 1, 3 and unrestricted depth were compared with AdaBoost, which sequentially trains weak tree learners on re-weighted samples and combines them by a weighted vote; Random Forest was included as a bagging reference. Models were trained on a stratified 80:20 split (random_state = 42) and additionally assessed with repeated stratified 5-fold cross-validation. Performance was measured with accuracy, precision, recall, F1-score and ROC-AUC. `[One or two sentences stating what the results showed, with numbers.]` `[One sentence stating the conclusion and its limits.]`

## 3. Introduction (½–1 page)
Cover: advertising and prediction; what machine learning offers; limitations of a single model; the idea of ensembles; what AdaBoost is; contribution of this report (an experimental comparison, not a new algorithm); outline of the report.

## 4. Problem Statement
Use the academic or report version from Part 1 of this guide. Add the central question, verbatim.

## 5. Real-World Application
The scenario, the uses (targeting, segmentation, campaign optimisation, personalisation, cost reduction), and the caveat that the model estimates likelihood and doesn't predict individuals with certainty. Mention that real systems also face privacy and fairness concerns when using personal attributes such as age or income. (One sentence is enough, but it shows maturity.)

## 6. Dataset Description
- Source and access date (cite the exact page).
- Table of columns (name, meaning, type, role).
- Statistics from *your* file: size, class distribution, missing values, duplicates, min/max/mean of Age and Salary, outlier check.
- Figures: purchase distribution, histograms, scatter.
- Statement that the dataset is educational and small.

## 7. Data Preprocessing
Pipeline in order: load → inspect → missing/duplicates → class balance → X/y → stratified 80:20 split → scaling check. Explain stratification, random_state, why trees don't need scaling (include your sanity-check result), and why the scaler must be fit on training data only.

## 8. Base Models
For each of stump, depth-3 tree, full tree (and Random Forest as a reference): description, key hyperparameters, strengths, weaknesses. Include the decision-tree concepts: nodes, Gini, entropy, information gain, depth, overfitting, pruning, with the Gini worked example from the guide, labelled hypothetical. Include your real depth-3 tree plot.

## 9. Ensemble Learning
Definition; why ensembles help (statistical, variance reduction, bias reduction); diversity requirement; bagging vs boosting vs stacking; the comparison table (Decision Tree / Random Forest / AdaBoost).

## 10. AdaBoost Methodology
Step-by-step algorithm; flow diagram; the toy example (labelled hypothetical); role of weak learners and why stumps are used; hyperparameters (`estimator`, `n_estimators`, `learning_rate`); scikit-learn implementation note (SAMME, no ½ factor).

## 11. Mathematical Foundation
The four equations with every symbol defined; the behaviour table for err near 0, = 0.5, > 0.5; assumptions (labels ±1, normalised weights, err < 0.5); optional one-paragraph remark on exponential loss and noise sensitivity.

## 12. Implementation
Environment (Python version, scikit-learn version — print `sklearn.__version__`), libraries, pipeline diagram, the exact model definitions and hyperparameters, evaluation protocol (hold-out + repeated CV), optional tuning procedure. Put key code in an appendix, not in the body. State that all code is reproducible with `random_state = 42`.

## 13. Evaluation Metrics
Definitions and formulas for TP/TN/FP/FN, accuracy, precision, recall, F1, ROC-AUC; why accuracy alone is insufficient; the business meaning of FP vs FN.

## 14. Experimental Results
Tables A–D (hold-out, CV, confusion counts, tuned), confusion-matrix figure, decision-boundary figure, AdaBoost accuracy-vs-rounds figure, errors/α figure. **Describe only what the tables and figures show.** Quote exact values from `results/*.csv`.

## 15. Comparison
Head-to-head: best single tree vs AdaBoost (hold-out and CV); AdaBoost vs Random Forest (bagging vs boosting); stump vs AdaBoost-with-stumps (the cleanest demonstration of "weak learner → strong learner"). Include the accuracy and PR/F1 bar charts, ROC curves and the CV boxplot. State differences in *numbers of test samples* as well as percentage points.

## 16. Discussion
Interpretation only. Cover: did the ensemble help, by how much, relative to run-to-run variability; precision/recall trade-off; overfitting evidence (train vs test; accuracy vs rounds); why the result may have happened (two features, overlapping classes, small data); sensitivity to split (CV); what the boundary plots reveal; honest caveats. Use hedged language ("suggests", "is consistent with") unless a result is clear.

## 17. Conclusion
Answer the central question in one paragraph using the template that matches your data (see Part 1 guide, section 13). Summarise key lessons about boosting. No new results here.

## 18. Future Scope
See the list below.

## 19. References
See below. Appendices: full code; extra figures; hyperparameter grid.

---

# Limitations (Part 25) — to put in the report and slide 16

1. **Small, simple dataset.** Typically 400 rows and two features; there is little room for any model to differ greatly.
2. **Educational / likely synthetic nature.** Provenance is unclear; results illustrate methods, not real customers.
3. **Limited features.** No browsing behaviour, price, product type or timing; so high accuracy shouldn't be read as high real-world predictability.
4. **Small test set.** With ≈ 80 test samples, each prediction is worth ≈ 1.25 percentage points; hold-out results are noisy. This is why CV is included.
5. **Split dependence.** A different `random_state` could change hold-out rankings.
6. **Possible overfitting.** Particularly for the full tree and for AdaBoost with many rounds or deeper learners; check the accuracy-vs-rounds plot.
7. **Noise/outlier sensitivity of AdaBoost.** Mislabelled or atypical points keep gaining weight.
8. **Limited generalisation.** Historical behaviour may not transfer to other platforms, products or periods; there is no external validation set.
9. **Fairness and privacy.** Using age/income for targeting can raise ethical concerns (brief mention).
10. **Hyperparameters.** Only a limited search was done (if you ran tuning).

(Don't exaggerate them — state each in one line and say how you mitigated it: stratification, CV, tuning on training data only.)

---

# Future Scope (Part 26)

All items stay within your assigned topic (boosting over trees) or are clearly framed as extensions.

- More customer features (gender, region, past behaviour, device) and a larger, real dataset.
- Cross-validated hyperparameter search for AdaBoost (depth, `n_estimators`, `learning_rate`), possibly with nested CV.
- Feature engineering (e.g. salary bands, age groups, interaction terms).
- Probability-based purchase prediction with calibration, for ranking customers and choosing thresholds by cost.
- Cost-sensitive evaluation using actual ad cost and sale margin.
- Comparison with **Gradient Boosting, XGBoost, LightGBM** as the natural next boosting methods (future work only; don't present results for them).
- Explainability (feature importances, SHAP) to show *why* a user is predicted to buy.
- Deployment as a small web application with real-time predictions.
- Customer segmentation using the leaf regions or predicted probabilities.
- Robustness study: add label noise and measure how AdaBoost's performance changes compared with a single tree.

---

# References (Part 27)

**Citation style: IEEE-like numbered list.** I have included only works I'm confident exist. Before submission, open each link/DOI once and confirm the details; add the access date for web pages.

**Core papers**
1. Y. Freund and R. E. Schapire, "A decision-theoretic generalization of on-line learning and an application to boosting," *Journal of Computer and System Sciences*, vol. 55, no. 1, pp. 119–139, 1997. DOI: 10.1006/jcss.1997.1504.
2. T. Hastie, S. Rosset, J. Zhu and H. Zou, "Multi-class AdaBoost," *Statistics and Its Interface*, vol. 2, no. 3, pp. 349–360, 2009. DOI: 10.4310/SII.2009.v2.n3.a8. *(The SAMME algorithm used by scikit-learn.)*
3. J. Friedman, T. Hastie and R. Tibshirani, "Additive logistic regression: a statistical view of boosting," *The Annals of Statistics*, vol. 28, no. 2, pp. 337–407, 2000. DOI: 10.1214/aos/1016218223. *(Optional: exponential-loss view.)*
4. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. DOI: 10.1023/A:1010933404324.
5. L. Breiman, "Bagging predictors," *Machine Learning*, vol. 24, no. 2, pp. 123–140, 1996. DOI: 10.1007/BF00058655.
6. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
7. T. G. Dietterich, "Ensemble methods in machine learning," in *Multiple Classifier Systems (MCS 2000)*, Lecture Notes in Computer Science, vol. 1857, Springer, 2000, pp. 1–15.

**Books**
8. T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009. (Chapters 9–10 for trees and boosting; Chapter 15 for random forests.)
9. G. James, D. Witten, T. Hastie and R. Tibshirani, *An Introduction to Statistical Learning*, 2nd ed. Springer, 2021. (Chapter 8: tree-based methods, bagging, boosting.)
10. L. Breiman, J. Friedman, R. Olshen and C. Stone, *Classification and Regression Trees*. Wadsworth, 1984.
11. A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, 3rd ed. O'Reilly Media, 2022.

**Official documentation** (add "accessed: DD Month 2026")
12. scikit-learn, "AdaBoostClassifier," https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
13. scikit-learn User Guide, "Ensembles: Gradient boosting, random forests, bagging, voting, stacking," https://scikit-learn.org/stable/modules/ensemble.html
14. scikit-learn, "DecisionTreeClassifier" and User Guide "Decision Trees," https://scikit-learn.org/stable/modules/tree.html
15. scikit-learn User Guide, "Metrics and scoring: quantifying the quality of predictions," https://scikit-learn.org/stable/modules/model_evaluation.html

**Dataset**
16. *Social Network Ads* dataset, `[exact page title, author/uploader if shown, site, URL, access date]` — cite the page you downloaded from; don't cite a different source than the one you actually used.

*(I deliberately did not invent a dataset author or URL; the dataset is widely mirrored and its original provenance is unclear.)*

---

# Final ISE Requirement Checklist (Part 30)

Legend: **[✓]** planned and fully covered by the materials here · **[→]** needs *your* data/execution/entry before it is truly complete.

| Requirement | Status | Where | What's still needed |
|---|---|---|---|
| Problem statement | [✓] | Slide 2; report §4 | — |
| Real-world application | [✓] | Slide 3; report §5 | — |
| Dataset | [→] | Slide 4; report §6 | Fill N, class split, ranges, source URL from your CSV |
| Input features | [✓] | Slide 4 | Confirm column names in your file |
| Target / output | [✓] | Slide 4 | Confirm 0/1 encoding from source |
| Individual ML models | [✓] | Slides 6–7; report §8 | — |
| Ensemble technique | [✓] | Slides 8–9; report §9–10 | — |
| Working of the ensemble | [✓] | Slides 9–10; report §10–11 | Build the animated flow diagram |
| Implementation methodology | [✓] | Slide 11; report §12 | Record your Python/scikit-learn versions |
| Individual model performance | [→] | Slide 13; report §14 | **Run the script; paste Table A/B/C** |
| Ensemble model performance | [→] | Slide 14; report §14 | **Same** |
| Comparison | [→] | Slide 15; report §15 | Written after results |
| Analysis | [→] | Slide 15; report §16 | Written after results (send me the output) |
| Conclusion | [→] | Slide 16; report §17 | Depends on results |
| References | [✓] | Slide 17; report §19 | Verify DOIs/links; add access dates; add dataset page |

**Still missing (cannot be done without you)**
1. Download the CSV and run the script.
2. Fill every `[ ]` placeholder in the slides from `results/*.csv`.
3. Confirm slide/time/report-format limits and the deadline with faculty (not in the ISE document).
4. Confirm the exact dataset source.
5. Build the visuals (flow diagram, animations) in PowerPoint/Canva.
6. Rehearse with a timer.

**Rubric self-check**
- *Understanding of the topic:* can each member explain bagging vs boosting, α, weight update, and why the test set is small? (Viva file.)
- *Quality of presentation:* ≤ 5 bullets per slide, charts readable from the back of the room, consistent colours.
- *Report quality:* every figure/table numbered and referred to in text; claims match numbers.
- *Timely submission:* confirm the deadline; submit early.
