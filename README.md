# Classification for Deception Detection
Task: Classify real or fake reviews.

Input: A corpus with manually analysed and annotated reviews.

Classifier used: Support Vector Machine

Metric evaluation: Accuracy

**ex1_Q1 to Q3.py:**
- parse the input file and return the text and label
- toFeatureVector will return a dictionary with key as features and the weights assigned to the feature, which can be either the frequency of the feature in the document or add more weights to specific words and so on. Binary feature can be used as well, if the feature exists then 1 else 0.
- 10 fold cross validation on the training data, and store the precision,recall, f1 score and accuracy

**ex1 - Q4 & Q5.py:**
- improved preprocessing to achieve better scores
- Additional features added for further improvement




