PS C:\Users\Pratham Mendon\OneDrive\Desktop\TE_BE_Codes\Sem-5\ML\ISE> & "C:\Program Files\Python313\python.exe" "c:/Users/Pratham Mendon/OneDrive/Desktop/TE_BE_Codes/Sem-5/ML/ISE/g15_adaboost_social_network_ads.py"

======================================================================
1. LOAD AND INSPECT
======================================================================
First 5 rows:
    User ID  Gender  Age  EstimatedSalary  Purchased
0  15624510    Male   19            19000          0
1  15810944    Male   35            20000          0
2  15668575  Female   26            43000          0
3  15603246  Female   27            57000          0
4  15804002    Male   19            76000          0 

Shape (rows, columns): (400, 5)

Column data types:
User ID             int64
Gender             object
Age                 int64
EstimatedSalary     int64
Purchased           int64
dtype: object

Missing values per column:
User ID            0
Gender             0
Age                0
EstimatedSalary    0
Purchased          0
dtype: int64

Duplicate rows (all columns)            : 0
Duplicate rows (features + target only) : 33

Summary statistics (feature ranges):
                 count        mean           std      min       25%      50%      75%       max
Age              400.0     37.6550     10.482877     18.0     29.75     37.0     46.0      60.0
EstimatedSalary  400.0  69742.5000  34096.960282  15000.0  43000.00  70000.0  88000.0  150000.0
Purchased        400.0      0.3575      0.479864      0.0      0.00      0.0      1.0       1.0

Target values present: [np.int64(0), np.int64(1)]

Class distribution:
           count  percent
Purchased                
0            257    64.25
1            143    35.75
Majority-class share (the accuracy of 'always predict the majority'): 64.25%

Potential outliers by the 1.5 x IQR rule:
  Age: 0 values outside [5.4, 70.4]
  EstimatedSalary: 0 values outside [-24500.0, 155500.0]

======================================================================
2. EXPLORATORY PLOTS
======================================================================

======================================================================
3. TRAIN-TEST SPLIT (80:20, stratified, fixed random_state)
======================================================================
Train size: 320 | Test size: 80
Class % in train: {0: 64.38, 1: 35.62}
Class % in test : {0: 63.75, 1: 36.25}
NOTE: with 80 test samples, ONE sample changes accuracy by 1.25 percentage points. Small differences between models may be noise -> see the repeated cross-validation in section 9.

======================================================================
4. FEATURE SCALING SANITY CHECK
======================================================================
Test predictions that differ between raw-feature and scaled-feature Decision Tree: 0
Raw accuracy   : 0.8625
Scaled accuracy: 0.8625
Trees split on thresholds of ONE feature at a time, so a monotonic rescaling does not change which samples fall in which region.
-> All main experiments below use the RAW (unscaled) features, which also keeps thresholds readable (e.g. Age <= 36.5).

======================================================================
5. MODELS
======================================================================
  [Baseline         ] Majority baseline
  [Individual tree  ] Decision Stump (depth 1)
  [Individual tree  ] Shallow Tree (depth 3)
  [Individual tree  ] Full Decision Tree
  [Bagging reference] Random Forest (bagging ref.)
  [Boosting         ] AdaBoost (stumps, 50)
  [Boosting         ] AdaBoost (depth-2 trees, 50)

======================================================================
6. HOLD-OUT TEST RESULTS (single 80:20 split)
======================================================================
                                          Group  Train Acc  Test Acc  Precision  Recall      F1  ROC-AUC
Model                                                                                                   
Majority baseline                      Baseline     0.6438    0.6375     0.0000  0.0000  0.0000   0.5000
Decision Stump (depth 1)        Individual tree     0.8312    0.8625     0.8750  0.7241  0.7925   0.8327
Shallow Tree (depth 3)          Individual tree     0.9281    0.8875     0.8125  0.8966  0.8525   0.9216
Full Decision Tree              Individual tree     0.9938    0.8625     0.7647  0.8966  0.8254   0.8698
Random Forest (bagging ref.)  Bagging reference     0.9938    0.9000     0.8387  0.8966  0.8667   0.9266
AdaBoost (stumps, 50)                  Boosting     0.9281    0.8625     0.8214  0.7931  0.8070   0.9327
AdaBoost (depth-2 trees, 50)           Boosting     0.9469    0.8750     0.8276  0.8276  0.8276   0.9490

Confusion matrices (rows = actual 0/1, columns = predicted 0/1):
  Majority baseline              TN=51  FP=0   FN=29  TP=0  
  Decision Stump (depth 1)       TN=48  FP=3   FN=8   TP=21 
  Shallow Tree (depth 3)         TN=45  FP=6   FN=3   TP=26 
  Full Decision Tree             TN=43  FP=8   FN=3   TP=26 
  Random Forest (bagging ref.)   TN=46  FP=5   FN=3   TP=26 
  AdaBoost (stumps, 50)          TN=46  FP=5   FN=6   TP=23 
  AdaBoost (depth-2 trees, 50)   TN=46  FP=5   FN=5   TP=24 

Detailed report for AdaBoost (stumps):
              precision    recall  f1-score   support

 No purchase       0.88      0.90      0.89        51
   Purchased       0.82      0.79      0.81        29

    accuracy                           0.86        80
   macro avg       0.85      0.85      0.85        80
weighted avg       0.86      0.86      0.86        80


======================================================================
7. COMPARISON PLOTS
======================================================================

======================================================================
8. DECISION BOUNDARIES AND ADABOOST INTERNALS
======================================================================
First 5 boosting rounds (AdaBoost with stumps):
 round  weighted_error  alpha (sklearn)  ln((1-err)/err)
     1          0.1688           1.5945           1.5945
     2          0.1830           1.4964           1.4964
     3          0.3169           0.7680           0.7680
     4          0.3775           0.5001           0.5001
     5          0.4054           0.3828           0.3828
For 2 classes scikit-learn's alpha equals learning_rate * ln((1-err)/err).
The textbook formula has an extra factor 1/2; this only rescales all alphas equally, so the final weighted vote has the same sign.

======================================================================
9. REPEATED STRATIFIED 5-FOLD CROSS-VALIDATION (5 repeats = 25 folds)
======================================================================
                                          Group         accuracy        precision           recall               f1          roc_auc
Model                                                                                                                               
Majority baseline                      Baseline  0.642 +/- 0.006  0.000 +/- 0.000  0.000 +/- 0.000  0.000 +/- 0.000  0.500 +/- 0.000
Decision Stump (depth 1)        Individual tree  0.831 +/- 0.038  0.839 +/- 0.081  0.659 +/- 0.066  0.736 +/- 0.061  0.793 +/- 0.042
Shallow Tree (depth 3)          Individual tree  0.902 +/- 0.034  0.838 +/- 0.067  0.909 +/- 0.051  0.870 +/- 0.044  0.931 +/- 0.039
Full Decision Tree              Individual tree  0.856 +/- 0.032  0.801 +/- 0.058  0.803 +/- 0.048  0.800 +/- 0.041  0.845 +/- 0.033
Random Forest (bagging ref.)  Bagging reference  0.886 +/- 0.033  0.833 +/- 0.055  0.859 +/- 0.056  0.844 +/- 0.044  0.935 +/- 0.033
AdaBoost (stumps, 50)                  Boosting  0.892 +/- 0.028  0.874 +/- 0.061  0.821 +/- 0.047  0.845 +/- 0.038  0.952 +/- 0.026
AdaBoost (depth-2 trees, 50)           Boosting  0.905 +/- 0.038  0.866 +/- 0.067  0.874 +/- 0.058  0.868 +/- 0.051  0.954 +/- 0.026

======================================================================
10. OPTIONAL HYPERPARAMETER TUNING (training data only)
======================================================================
Best parameters (by 5-fold CV F1 on train set): {'estimator__max_depth': 2, 'learning_rate': 0.1, 'n_estimators': 25}
Best CV F1 on train set: 0.8872

Tuned AdaBoost evaluated ONCE on the test set:
  Accuracy : 0.9
  Precision: 0.8387
  Recall   : 0.8966
  F1       : 0.8667
  ROC-AUC  : 0.9341

======================================================================
11. ADABOOST FROM SCRATCH (teaching demo, 10 rounds)
======================================================================
Round  1: weighted error = 0.1687 | alpha = 0.7973 | max sample weight = 0.0093
Round  2: weighted error = 0.1830 | alpha = 0.7482 | max sample weight = 0.0253
Round  3: weighted error = 0.3169 | alpha = 0.3840 | max sample weight = 0.0399
Round  4: weighted error = 0.3775 | alpha = 0.2500 | max sample weight = 0.0321
Round  5: weighted error = 0.4054 | alpha = 0.1914 | max sample weight = 0.0395
Round  6: weighted error = 0.4185 | alpha = 0.1645 | max sample weight = 0.0340
Round  7: weighted error = 0.4299 | alpha = 0.1411 | max sample weight = 0.0395
Round  8: weighted error = 0.4385 | alpha = 0.1236 | max sample weight = 0.0352
Round  9: weighted error = 0.4515 | alpha = 0.0973 | max sample weight = 0.0321
Round 10: weighted error = 0.3905 | alpha = 0.2225 | max sample weight = 0.0411

From-scratch AdaBoost test accuracy: 0.8625
(Only to show the mechanism - use the scikit-learn model for reported results.)

======================================================================
DONE - figures in ./figures, tables in ./results
======================================================================
PS C:\Users\Pratham Mendon\OneDrive\Desktop\TE_BE_Codes\Sem-5\ML\ISE> 