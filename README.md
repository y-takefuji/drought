# drought
Download the original train dataset from the following site:

https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/data

Run train.py to generate train.csv

Run rf.py to conduct 10-fold cross-validation with random forest.
<pre>
Average accuracy: 0.6366 (±0.0021)

Confusion Matrix:
[[29676  6031  1939  1009   402   167]
 [ 8511 20067  7044  2339   942   321]
 [ 3139  7003 19166  7083  2189   644]
 [ 1198  1772  6087 21333  7512  1322]
 [  394   434  1131  5478 25361  6426]
 [   26    42   101   387  4442 34226]]
Class labels: [0, 1, 2, 3, 4, 5]
</pre>
