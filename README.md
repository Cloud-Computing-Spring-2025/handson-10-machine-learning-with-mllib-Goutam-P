# Handson-10-MachineLearning-with-MLlib

# Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

##  Tasks

###  Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Output Example:**
```
+--------------------+-----+
|features            |label|
+--------------------+-----+
|(15,[0,1,5,10,...]) |  0.0|
|(15,[0,3,6,12,...]) |  1.0|
+--------------------+-----+
```

---

###  Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Code Output:**
```
 Logistic Regression AUC: 0.7531
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Save the selected feature vectors.

**Saved Output:**
```
 output/chi_square_selected_features.csv
```

---

###  Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output Example:**
```
 Tuning Logistic Regression...
 Logistic Regression Best AUC: 0.7575

 Tuning Decision Tree...
 Decision Tree Best AUC: 0.7187

 Tuning Random Forest...
 Random Forest Best AUC: 0.8311

 Tuning Gradient Boosted Trees...
 GBT Best AUC: 0.7672
```

**Saved Output:**
```
 output/model_auc_results.csv
 output/final_predictions.csv
```

---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` and `pandas` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

```bash
# Install pyspark and pandas if not installed
pip install pyspark pandas

# Run the script with spark-submit
spark-submit customer_churn_analysis.py
```

**OUTPUT**

python3 customer-churn-analysis.py
25/04/08 17:09:45 WARN Utils: Your hostname, codespaces-4715db resolves to a loopback address: 127.0.0.1; using 10.0.4.26 instead (on interface eth0)
25/04/08 17:09:45 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/04/08 17:09:45 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
25/04/08 17:09:58 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
25/04/08 17:09:58 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.VectorBLAS
✅ Logistic Regression AUC: 0.7531
+----------------------+-----+
|selectedFeatures      |label|
+----------------------+-----+
|[1.0,0.0,0.0,1.0,36.0]|0.0  |
|[1.0,0.0,0.0,0.0,13.0]|0.0  |
|[1.0,0.0,0.0,0.0,1.0] |0.0  |
|[0.0,1.0,0.0,1.0,1.0] |0.0  |
|[1.0,0.0,0.0,1.0,4.0] |0.0  |
|[1.0,0.0,0.0,0.0,12.0]|0.0  |
|[1.0,0.0,0.0,1.0,37.0]|0.0  |
|[1.0,0.0,1.0,0.0,50.0]|0.0  |
|[1.0,0.0,1.0,0.0,29.0]|0.0  |
|[0.0,1.0,1.0,0.0,6.0] |0.0  |
|[1.0,0.0,0.0,1.0,17.0]|1.0  |
|[1.0,0.0,0.0,0.0,14.0]|0.0  |
|[1.0,0.0,0.0,1.0,70.0]|1.0  |
|[0.0,1.0,1.0,0.0,14.0]|0.0  |
|(5,[0,2],[1.0,1.0])   |0.0  |
|[1.0,0.0,0.0,0.0,65.0]|0.0  |
|[1.0,0.0,0.0,1.0,12.0]|0.0  |
|[1.0,0.0,1.0,0.0,26.0]|0.0  |
|[1.0,0.0,0.0,1.0,11.0]|1.0  |
|[1.0,0.0,1.0,0.0,12.0]|0.0  |
+----------------------+-----+
only showing top 20 rows


🔍 Tuning Logistic Regression...
25/04/08 17:10:02 WARN GarbageCollectionMetrics: To enable non-built-in garbage collector(s) List(G1 Concurrent GC), users should configure it(them) to spark.eventLog.gcMetrics.youngGenerationGarbageCollectors or spark.eventLog.gcMetrics.oldGenerationGarbageCollectors
✅ Logistic Regression Best AUC: 0.7575
🏷️  Best Params: {Param(parent='LogisticRegression_c935b1c07499', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2).'): 2, Param(parent='LogisticRegression_c935b1c07499', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0, Param(parent='LogisticRegression_c935b1c07499', name='family', doc='The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial'): 'auto', Param(parent='LogisticRegression_c935b1c07499', name='featuresCol', doc='features column name.'): 'features', Param(parent='LogisticRegression_c935b1c07499', name='fitIntercept', doc='whether to fit an intercept term.'): True, Param(parent='LogisticRegression_c935b1c07499', name='labelCol', doc='label column name.'): 'label', Param(parent='LogisticRegression_c935b1c07499', name='maxBlockSizeInMB', doc='maximum memory in MB for stacking input data into blocks. Data is stacked within partitions. If more than remaining data size in a partition then it is adjusted to the data size. Default 0.0 represents choosing optimal value, depends on specific algorithm. Must be >= 0.'): 0.0, Param(parent='LogisticRegression_c935b1c07499', name='maxIter', doc='max number of iterations (>= 0).'): 100, Param(parent='LogisticRegression_c935b1c07499', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='LogisticRegression_c935b1c07499', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='LogisticRegression_c935b1c07499', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='LogisticRegression_c935b1c07499', name='regParam', doc='regularization parameter (>= 0).'): 0.1, Param(parent='LogisticRegression_c935b1c07499', name='standardization', doc='whether to standardize the training features before fitting the model.'): True, Param(parent='LogisticRegression_c935b1c07499', name='threshold', doc='Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g. if threshold is p, then thresholds must be equal to [1-p, p].'): 0.5, Param(parent='LogisticRegression_c935b1c07499', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0).'): 1e-06}

🔍 Tuning Decision Tree...
✅ Decision Tree Best AUC: 0.7187
🏷️  Best Params: {Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='featuresCol', doc='features column name.'): 'features', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='labelCol', doc='label column name.'): 'label', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='leafCol', doc='Leaf indices column name. Predicted leaf index of each instance in each tree by preorder.'): '', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'): 5, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='minWeightFractionPerNode', doc='Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5).'): 0.0, Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='DecisionTreeClassifier_fb265c19cb2f', name='seed', doc='random seed.'): -3380985765543548593}

🔍 Tuning Random Forest...
✅ Random Forest Best AUC: 0.8311
🏷️  Best Params: {Param(parent='RandomForestClassifier_399297ecaf6c', name='bootstrap', doc='Whether bootstrap samples are used when building trees.'): True, Param(parent='RandomForestClassifier_399297ecaf6c', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False, Param(parent='RandomForestClassifier_399297ecaf6c', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10, Param(parent='RandomForestClassifier_399297ecaf6c', name='featureSubsetStrategy', doc="The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto'"): 'auto', Param(parent='RandomForestClassifier_399297ecaf6c', name='featuresCol', doc='features column name.'): 'features', Param(parent='RandomForestClassifier_399297ecaf6c', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini', Param(parent='RandomForestClassifier_399297ecaf6c', name='labelCol', doc='label column name.'): 'label', Param(parent='RandomForestClassifier_399297ecaf6c', name='leafCol', doc='Leaf indices column name. Predicted leaf index of each instance in each tree by preorder.'): '', Param(parent='RandomForestClassifier_399297ecaf6c', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32, Param(parent='RandomForestClassifier_399297ecaf6c', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'): 5, Param(parent='RandomForestClassifier_399297ecaf6c', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256, Param(parent='RandomForestClassifier_399297ecaf6c', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0, Param(parent='RandomForestClassifier_399297ecaf6c', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1, Param(parent='RandomForestClassifier_399297ecaf6c', name='minWeightFractionPerNode', doc='Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5).'): 0.0, Param(parent='RandomForestClassifier_399297ecaf6c', name='numTrees', doc='Number of trees to train (>= 1).'): 20, Param(parent='RandomForestClassifier_399297ecaf6c', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='RandomForestClassifier_399297ecaf6c', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='RandomForestClassifier_399297ecaf6c', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='RandomForestClassifier_399297ecaf6c', name='seed', doc='random seed.'): 2810597348954768666, Param(parent='RandomForestClassifier_399297ecaf6c', name='subsamplingRate', doc='Fraction of the training data used for learning each decision tree, in range (0, 1].'): 1.0}

🔍 Tuning Gradient Boosted Trees...
✅ Gradient Boosted Trees Best AUC: 0.7672
🏷️  Best Params: {Param(parent='GBTClassifier_a16f16c75532', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False, Param(parent='GBTClassifier_a16f16c75532', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10, Param(parent='GBTClassifier_a16f16c75532', name='featureSubsetStrategy', doc="The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto'"): 'all', Param(parent='GBTClassifier_a16f16c75532', name='featuresCol', doc='features column name.'): 'features', Param(parent='GBTClassifier_a16f16c75532', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: variance'): 'variance', Param(parent='GBTClassifier_a16f16c75532', name='labelCol', doc='label column name.'): 'label', Param(parent='GBTClassifier_a16f16c75532', name='leafCol', doc='Leaf indices column name. Predicted leaf index of each instance in each tree by preorder.'): '', Param(parent='GBTClassifier_a16f16c75532', name='lossType', doc='Loss function which GBT tries to minimize (case-insensitive). Supported options: logistic'): 'logistic', Param(parent='GBTClassifier_a16f16c75532', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32, Param(parent='GBTClassifier_a16f16c75532', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'): 5, Param(parent='GBTClassifier_a16f16c75532', name='maxIter', doc='max number of iterations (>= 0).'): 10, Param(parent='GBTClassifier_a16f16c75532', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256, Param(parent='GBTClassifier_a16f16c75532', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0, Param(parent='GBTClassifier_a16f16c75532', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1, Param(parent='GBTClassifier_a16f16c75532', name='minWeightFractionPerNode', doc='Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5).'): 0.0, Param(parent='GBTClassifier_a16f16c75532', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='GBTClassifier_a16f16c75532', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='GBTClassifier_a16f16c75532', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='GBTClassifier_a16f16c75532', name='seed', doc='random seed.'): 5114036666769305723, Param(parent='GBTClassifier_a16f16c75532', name='stepSize', doc='Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.'): 0.1, Param(parent='GBTClassifier_a16f16c75532', name='subsamplingRate', doc='Fraction of the training data used for learning each decision tree, in range (0, 1].'): 1.0, Param(parent='GBTClassifier_a16f16c75532', name='validationTol', doc='Threshold for stopping early when fit with validation is used. If the error rate on the validation input changes by less than the validationTol, then learning will stop early (before `maxIter`). This parameter is ignored when fit without validation is used.'): 0.01}