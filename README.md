# drought
Download the original train dataset from the following site:

https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/data

Run train.py to generate train.csv

Run rf.py to conduct 10-fold cross-validation with random forest.
<pre>
Average CV accuracy: 0.6366 (±0.0021)

Confusion Matrix:
[[29676  6031  1939  1009   402   167]
 [ 8511 20067  7044  2339   942   321]
 [ 3139  7003 19166  7083  2189   644]
 [ 1198  1772  6087 21333  7512  1322]
 [  394   434  1131  5478 25361  6426]
 [   26    42   101   387  4442 34226]]
Class labels: [0, 1, 2, 3, 4, 5]
</pre>
Run cv5.py to conduct 5-fold cross-validation with random forest with top 5 features.
<pre>
Top 5 features by PCA contribution:
1. WS50M_RANGE: 1.1881
2. WS50M_MIN: 1.1361
3. WS10M_MIN: 1.1179
4. WS10M_RANGE: 1.0814
5. T2M_RANGE: 1.0336

Conducting 5-fold cross-validation with Random Forest using PCA top 5 features...
PCA - 5-fold CV Accuracy: 0.7396 ± 0.0007

Top 5 features by Spearman correlation:
1. T2M_RANGE: 0.2930
2. T2M_MAX: 0.2152
3. PS: 0.2065
4. TS: 0.1704
5. T2M: 0.1603

Conducting 5-fold cross-validation with Random Forest using Spearman reduced dataset...
Spearman - 5-fold CV Accuracy: 0.8155 ± 0.0020

Top 5 features by Kendall correlation:
1. T2M_RANGE: 0.2394
2. T2M_MAX: 0.1757
3. PS: 0.1687
4. TS: 0.1391
5. T2M: 0.1309

Conducting 5-fold cross-validation with Random Forest using Kendall reduced dataset...
Kendall - 5-fold CV Accuracy: 0.8155 ± 0.0020
</pre>
