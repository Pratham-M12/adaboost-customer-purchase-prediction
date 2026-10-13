# ISE 2 — Group G15 — Viva Preparation (Part 4 of 4)

Deliverables 18 and 19. Answers are written to be *spoken*: short first sentence, then one or two supporting sentences. Where an answer needs your own numbers, it says `[your result]` — fill it in from your run, never guess.

Numbering: **Q1–Q30** follow your list; **Q31–Q40** are extras; **X1–X11** are the harder cross-questions.

---

## Core questions (Q1–Q30)

**Q1. What is Ensemble Learning?**
Combining several models so the group predicts better than a typical single model. The members' mistakes partly cancel or are corrected by other members. It works when members are better than random and make different errors.

**Q2. Why did you use AdaBoost?**
It was our assigned technique. It is also the classic boosting method: it shows clearly how weak learners are combined using sample re-weighting and a weighted vote, which is the main focus of the ISE.

**Q3. Difference between bagging and boosting?**
Bagging trains models in parallel on random bootstrap samples and averages them; it mainly reduces variance (Random Forest). Boosting trains models one after another, each focusing on earlier mistakes, and combines them with a weighted vote; it mainly reduces bias (AdaBoost).

**Q4. What is a weak learner?**
A model only slightly better than random guessing: error below 0.5 on the weighted data. In our project it's a decision tree of depth 1 (a stump), or at most depth 2–3.

**Q5. Why are decision stumps useful in AdaBoost?**
They are simple, fast, low-variance and can't memorise the data. Because each stump is weak, the sequence has room to improve, and re-weighting makes each new stump different from the last, so we get diversity.

**Q6. Why does AdaBoost increase the weight of misclassified samples?**
So the next learner is forced to concentrate on the cases the ensemble currently gets wrong. Mathematically it is a greedy way of minimising exponential loss, which penalises mistakes heavily.

**Q7. What is the role of alpha?**
It's the learner's voting power: α = ½ ln((1 − err)/err). Low error gives high α, error 0.5 gives zero, so accurate learners count more in the final vote. It also sets how strongly the sample weights are updated.

**Q8. What happens if the weak learner has error greater than 0.5?**
Then it's worse than random, and the formula gives a negative α, which would flip its vote. In scikit-learn (SAMME) such a learner is discarded and boosting stops. The "weak learner assumption" requires error below 0.5.

**Q9. Why is this a classification problem?**
The output is a category: purchased (1) or not purchased (0). We predict a class, not a continuous quantity.

**Q10. What is the target variable?**
`Purchased`, a binary label: 0 = did not purchase, 1 = purchased (the usual encoding; confirm from the dataset source).

**Q11. Why do we split the dataset?**
To estimate performance on data the model hasn't seen. Evaluating on training data would reward memorisation. We train on 80% and test on 20%.

**Q12. Why do we use stratification?**
To keep the class proportion the same in train and test. Our classes are not 50/50 (`[x]`% purchased), so a plain random split could shift the proportions and make comparison unfair.

**Q13. What is overfitting?**
When a model learns noise and quirks of the training set, so training accuracy is high but test accuracy is lower. The gap between the two is the symptom.

**Q14. How can a decision tree overfit?**
If we allow unlimited depth, it keeps splitting until each leaf is pure, effectively memorising individual customers. We saw / expect `[Train Acc vs Test Acc for your full tree]`. We control it with `max_depth` or pruning.

**Q15. Why does Random Forest differ from AdaBoost?**
Random Forest trains many deep trees independently on bootstrap samples with random feature subsets and takes an equal vote. AdaBoost trains shallow trees sequentially on re-weighted data and takes a weighted vote. One reduces variance by averaging; the other reduces bias by correcting mistakes.

**Q16. What is Gini impurity?**
A measure of how mixed the classes are in a node: Gini = 1 − Σpₖ². It is 0 for a pure node and 0.5 for a 50/50 binary node. Trees choose the split that lowers it the most.

**Q17. What is entropy?**
Another impurity measure: H = −Σpₖ log₂ pₖ. It's 0 for a pure node and 1 bit for a 50/50 binary node. It measures uncertainty.

**Q18. What is information gain?**
The reduction in impurity from a split: impurity of the parent minus the weighted impurity of the children. The tree picks the split with the highest gain.

**Q19. What is accuracy?**
(TP + TN) / total predictions: the fraction predicted correctly.

**Q20. What is precision?**
TP / (TP + FP): of the users we predicted would buy, the fraction who actually did. High precision means few wasted ads.

**Q21. What is recall?**
TP / (TP + FN): of the users who actually bought, the fraction we found. High recall means few missed customers.

**Q22. What is F1-score?**
The harmonic mean of precision and recall, 2PR/(P + R). It is high only when both are reasonably high.

**Q23. What is a confusion matrix?**
A 2×2 table of actual vs predicted classes: TN, FP, FN, TP. It shows *which* mistakes a model makes, not just how many.

**Q24. What is a false positive?**
We predict "will purchase" but the user doesn't: a wasted ad impression.

**Q25. What is a false negative?**
We predict "won't purchase" but the user would have: a missed sale.

**Q26. Why should we not rely only on accuracy?**
Accuracy hides the type of error and can look good under class imbalance; always predicting the majority class already scores `[majority-class %]`. Precision, recall, F1 and the confusion matrix show the trade-off between wasted ads and lost customers.

**Q27. Does AdaBoost always improve accuracy?**
No. It often helps when a single simple model underfits, but on a small, easy or noisy dataset the gain can be small or absent. Our result: `[state your finding, with numbers and CV spread]`.

**Q28. What are the limitations of AdaBoost?**
Sequential training can't be parallelised across rounds; it is sensitive to noisy labels and outliers because their weights keep growing; it needs a weak learner that beats random; and it can overfit with many rounds or complex learners.

**Q29. What happens when the dataset contains noisy data or outliers?**
Mislabelled or atypical points keep being misclassified, so their weights grow and later learners spend effort on them, which can hurt generalisation. Mitigations: clean data, lower learning rate, fewer rounds, shallower learners, or more robust boosting variants.

**Q30. How would you improve this project?**
Add more features and a larger real dataset, tune hyperparameters with cross-validation, compare with gradient boosting/XGBoost, add explainability, calibrate probabilities, and evaluate with real ad-cost/margin numbers.

---

## Extra questions (Q31–Q40)

**Q31. Why did you drop User ID and not use Gender?**
User ID is just an identifier with no predictive meaning. Gender is optional; the main experiment uses only Age and EstimatedSalary, as is standard, which also lets us plot decision boundaries in 2-D. Adding Gender is listed as a possible extension.

**Q32. Is your dataset balanced?**
No: about `[x]`% purchased (`[n0]` vs `[n1]`; from your file). It's mildly imbalanced, which is why we show precision, recall, F1 and use stratified splitting.

**Q33. What is ROC-AUC?**
The area under the curve of true-positive rate vs false-positive rate across all thresholds. It equals the probability that a random buyer gets a higher score than a random non-buyer. 0.5 is chance, 1.0 is perfect. A single tree gives coarse scores, so its curve has few points.

**Q34. What does `learning_rate` do in AdaBoost?**
It scales each learner's contribution. Smaller values make learning more cautious and usually need more estimators; there's a trade-off between `learning_rate` and `n_estimators`.

**Q35. What does `n_estimators` do?**
It's the maximum number of boosting rounds (weak learners). More rounds fit the training data more closely; the accuracy-vs-rounds plot shows when performance plateaus or the train–test gap opens.

**Q36. Why 80:20?**
It's a common compromise: enough data to train, enough to test. With only 400 rows the test set is small (`[n]` samples), so we also used repeated cross-validation.

**Q37. Why didn't you scale the features?**
Trees split on one-feature thresholds, so monotonic rescaling doesn't change which samples go left or right. We verified this: `[n]` prediction differences between raw and scaled.

**Q38. Is the dataset real? How reliable are conclusions?**
It's an educational dataset of unclear provenance. Our conclusions are about method behaviour on this dataset, not a claim about real customers.

**Q39. What is bias vs variance?**
Bias is error from a model that's too simple (a stump underfits). Variance is error from sensitivity to the particular training sample (a full tree overfits). Boosting mainly attacks bias; bagging mainly attacks variance.

**Q40. What is pruning?**
Reducing a tree's size to cut overfitting, either by stopping growth early (`max_depth`, `min_samples_leaf`) or by cutting back a fully grown tree (cost-complexity pruning, `ccp_alpha`).

---

## Harder cross-questions (X1–X11)

**X1. "If Random Forest already combines multiple trees, why do you need AdaBoost?"**
We don't "need" it; they work differently and neither dominates. Random Forest averages many deep, independent trees to cut variance. AdaBoost builds a sequence of shallow trees that correct each other's mistakes to cut bias. Which is better depends on the data, and that is exactly what our comparison tests; our result was `[state it]`.

**X2. "Is AdaBoost itself a decision tree?"**
No. AdaBoost is a meta-algorithm for combining learners. Here the learners are decision trees (stumps), but they could be other weak models. The final model is a weighted vote of many trees, not a tree.

**X3. "Why are the trees in AdaBoost sequential while Random Forest trees are independent?"**
In AdaBoost each tree depends on the sample weights, which depend on the errors of all earlier trees, so tree *t* can't be trained before tree *t − 1* finishes. In Random Forest each tree sees a random bootstrap sample and doesn't need the others, so they can be trained in parallel.

**X4. "Why not use XGBoost?"**
Our assigned technique is AdaBoost over tree learners, and the ISE aims to understand the combination mechanism. XGBoost is gradient boosting with regularisation and engineering optimisations; it is a natural future comparison, but we don't claim anything about its performance here.

**X5. "Why did you choose a shallow tree as the base learner?"**
A weak learner needs to beat random guessing but stay simple; shallow trees are low-variance and fast, so the sequence can improve gradually and each round has room to add something. Deeper trees can fit the training data alone and overfit. We also tested depth-2 trees to see the effect: `[your finding]`.

**X6. "Does feature scaling affect Decision Trees?"**
No, not their predictions: splits compare a single feature with a threshold, and monotonic transforms preserve ordering. It would matter for k-NN, SVM, logistic regression or neural networks. Our script includes a check that confirms it.

**X7. "Can AdaBoost be used for regression?"**
Yes. AdaBoost.R2 (Drucker, 1997) adapts the idea to regression; scikit-learn offers `AdaBoostRegressor`. Our task is classification, so we used `AdaBoostClassifier`.

**X8. "What happens if one weak learner performs very poorly?"**
If its weighted error is near 0.5, α is near 0, so it barely influences the vote. If error exceeds 0.5, scikit-learn discards it and stops. So a bad learner is automatically down-weighted, though a run of bad learners means boosting can't continue to help.

**X9. "Why does AdaBoost focus on misclassified samples?"**
Because the ones already classified correctly give little new information. Re-weighting forces the next learner to specialise on the hard region of the feature space, so the combined model covers areas the earlier learners miss. In the toy example, after one round the mistakes carry half of the total weight.

**X10. "What is the difference between model diversity and model accuracy?"**
Accuracy is how often one model is right; diversity is how differently the models err. An ensemble helps when members are individually better than chance *and* their mistakes don't coincide. Ten identical 90%-accurate models give 90%; ten 80% models with independent errors can do better than 80%. AdaBoost produces diversity through re-weighting; Random Forest through bootstrapping and random features.

**X11. "How do you know the ensemble actually improved the model?"**
We compare against the *best* single tree, not the weakest, using the same split. We express the gap in number of test samples (one sample ≈ `[100/n]` points). And we repeat evaluation over 25 stratified CV folds, comparing means and spreads: improvement counts as real only if it is larger than fold-to-fold variation. Our conclusion: `[state it honestly, including "no clear improvement" if that's what the data show]`.

---

## Rapid-fire formulas to memorise
- Gini = 1 − Σpₖ² · Entropy = −Σpₖ log₂pₖ · IG = H(parent) − Σ(nᵢ/n)H(childᵢ)
- err = Σwᵢ·I(yᵢ ≠ h(xᵢ)) · α = ½ln((1−err)/err) · wᵢ ← wᵢ·exp(−α yᵢ h(xᵢ)), normalise · H(x) = sign(Σαₜhₜ(x))
- Accuracy, Precision, Recall, F1 (see Slide 12).
- sklearn (2 classes): α_sk = learning_rate × ln((1−err)/err) → exactly 2× the textbook α when learning_rate = 1; same final prediction.

## Toy example to be able to redo from memory
10 samples → w = 0.1 · 3 mistakes → err 0.3 · α = 0.424 · new weights: mistakes 1/6 each, correct 1/14 each · mistakes now hold 0.5 of the weight.
