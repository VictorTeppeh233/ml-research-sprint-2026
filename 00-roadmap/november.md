# November 2026 — Classical Machine Learning & Statistical Foundations

> **Sprint:** ML Research Sprint 2026
> **Period:** November 1–30, 2026
> **Theme:** Probability → Optimization → Classical ML → Evaluation → Ensembles → Dimensionality Reduction
> **Level:** Beginner → Intermediate
> **Primary objective:** Develop a rigorous understanding of classical machine learning, including the mathematics behind important algorithms, their implementations, evaluation methods, and practical limitations.

---

## 1. November Objective

October established the foundations:

**Python → NumPy → Linear Algebra → Calculus → Probability → Linear Regression → Gradient Descent → Basic Classification**

November moves from foundational concepts into a more systematic study of machine learning.

By the end of November, the goal is to understand not only **how to use machine-learning algorithms**, but also:

* why they work
* what assumptions they make
* how their mathematics works
* how to implement simplified versions from scratch
* how to evaluate them correctly
* how to diagnose overfitting and underfitting
* how preprocessing affects models
* how to compare models experimentally
* how ensembles improve predictive performance
* how dimensionality reduction works
* how to communicate experimental results technically

The progression for November is:

> **Probability & Statistics → Optimization → Classification → Evaluation → Feature Engineering → Ensembles → Clustering → PCA → End-to-End ML Project**

---

# 2. Core Resources

## Primary resources

### Mathematics for Machine Learning

**Deisenroth, Faisal & Ong**

Primary chapters:

* Chapter 5 — Probability and Statistics
* Chapter 6 — Linear Regression
* Chapter 7 — Optimization

Use this book primarily for mathematical understanding.

---

### Hands-On Machine Learning

**Aurélien Géron**

Primary chapters:

* Chapter 1 — The Machine Learning Landscape
* Chapter 2 — End-to-End Machine Learning Project
* Chapter 3 — Classification
* Chapter 6 — Decision Trees
* Chapter 7 — Ensemble Learning and Random Forests
* Chapter 8 — Dimensionality Reduction

Use this as the primary practical ML resource.

---

### An Introduction to Statistical Learning

**James, Witten, Hastie, Tibshirani & Taylor**

Primary chapters:

* Chapter 2 — Statistical Learning
* Chapter 3 — Linear Regression
* Chapter 4 — Classification
* Chapter 5 — Resampling Methods
* Chapter 6 — Linear Model Selection and Regularization
* Chapter 8 — Tree-Based Methods
* Chapter 10 — Unsupervised Learning

ISLR should increasingly become an important part of the theoretical side of the sprint.

---

### Introduction to Applied Linear Algebra

**Boyd & Vandenberghe**

Use as a reference when studying:

* vectors
* inner products
* projections
* least squares
* eigenvalues
* eigenvectors
* PCA

Do not attempt to read the entire book sequentially during November.

---

## Reference documentation

Use official documentation when implementing models:

* NumPy
* pandas
* scikit-learn
* Matplotlib

Documentation is not a substitute for understanding the algorithm.

The workflow should be:

> **Understand → Read documentation → Implement → Experiment → Compare → Explain**

---

# 3. November Reading Map

| Date   | Mathematics for ML               | Hands-On ML                      | ISLR               | Practical Work                      |
| ------ | -------------------------------- | -------------------------------- | ------------------ | ----------------------------------- |
| Nov 1  | Ch. 5 — Probability & Statistics | Ch. 1–2 review                   | —                  | Probability exercises               |
| Nov 2  | Ch. 6 — Optimization             | —                                | —                  | Gradient descent                    |
| Nov 3  | Ch. 6 — Optimization             | —                                | —                  | Gradient descent + LaTeX derivation |
| Nov 4  | Ch. 5 — Probability              | —                                | —                  | Random variables & distributions    |
| Nov 5  | Ch. 5 — Statistics               | —                                | —                  | Variance, covariance, correlation   |
| Nov 6  | Ch. 5–6 review                   | Ch. 2                            | —                  | Generalization & overfitting        |
| Nov 7  | Ch. 6 — Optimization             | Ch. 2                            | Ch. 2              | Optimization + ML workflow          |
| Nov 8  | —                                | Ch. 3 — Classification           | Ch. 2              | MNIST classification                |
| Nov 9  | Probability review               | Ch. 3                            | Ch. 4              | Logistic regression                 |
| Nov 10 | —                                | Ch. 3                            | Ch. 4              | Logistic regression from scratch    |
| Nov 11 | —                                | Ch. 3                            | Ch. 4              | KNN from scratch                    |
| Nov 12 | —                                | Ch. 6 — Decision Trees           | Ch. 8              | Decision-tree fundamentals          |
| Nov 13 | Probability/statistics           | Ch. 6                            | Ch. 8              | Entropy & Gini + LaTeX              |
| Nov 14 | Ch. 5–6 review                   | Ch. 3 & 6                        | Ch. 4 & 8          | Compare classifiers                 |
| Nov 15 | Statistics                       | Ch. 3                            | Ch. 5              | Accuracy, precision, recall         |
| Nov 16 | Statistics                       | Ch. 2                            | Ch. 3              | MAE, MSE, RMSE, R²                  |
| Nov 17 | —                                | Ch. 2                            | Ch. 5              | Train/validation/test               |
| Nov 18 | —                                | Ch. 2                            | Ch. 5              | Cross-validation                    |
| Nov 19 | —                                | Ch. 2                            | Ch. 6              | Feature engineering                 |
| Nov 20 | —                                | Ch. 2                            | Ch. 6              | Pipelines & preprocessing           |
| Nov 21 | Statistics review                | Ch. 2–3                          | Ch. 5              | Model evaluation experiment         |
| Nov 22 | —                                | Ch. 7 — Ensemble Learning        | Ch. 8              | Voting & bagging                    |
| Nov 23 | —                                | Ch. 7                            | Ch. 8              | Random forests + paper              |
| Nov 24 | Ch. 6 review                     | Ch. 7                            | Ch. 8              | Gradient boosting                   |
| Nov 25 | —                                | Ch. 8 — Dimensionality Reduction | Ch. 10             | K-Means                             |
| Nov 26 | —                                | Ch. 8                            | Ch. 10             | K-Means from scratch                |
| Nov 27 | Linear algebra                   | Ch. 8                            | Ch. 10             | PCA mathematics                     |
| Nov 28 | Eigenvalues/eigenvectors         | Ch. 8                            | Ch. 10             | PCA with NumPy                      |
| Nov 29 | —                                | Ch. 2                            | Ch. 2–10 reference | End-to-end ML project               |
| Nov 30 | Full mathematical review         | Relevant chapters                | Relevant chapters  | Final project report                |

---

# 4. Week 1 — Probability, Statistics & Optimization

## November 1–7

### Goal

Build the mathematical tools needed to reason about uncertainty, data, loss functions, and optimization.

The objective is not to become a statistician in one week.

The objective is to understand the statistical concepts that machine-learning algorithms rely on.

---

## November 1 — Probability & Statistics

### Study

**Mathematics for Machine Learning — Chapter 5**

Focus on:

* probability
* sample spaces
* events
* conditional probability
* independence
* random variables
* expectation
* variance

### Practice

Implement basic probability experiments in Python.

Examples:

* coin flips
* dice rolls
* empirical probability
* expected value
* variance

### Deliverable

```text
02-mathematics/
└── probability/
    ├── notes/
    └── exercises/
        └── probability_simulations.ipynb
```

---

## November 2 — Optimization

### Study

**Mathematics for Machine Learning — Chapter 6**

Focus on:

* objective functions
* optimization
* local minima
* global minima
* gradients
* learning rate
* iterative optimization

### Implementation

Implement gradient descent for a simple one-dimensional function.

For example:

$$
f(x)=x^2
$$

The gradient is:

$$
\nabla f(x)=2x
$$

Update:

$$
x_{t+1}=x_t-\eta\nabla f(x_t)
$$

---

## November 3 — Gradient Descent Deep Dive

### Study

Continue optimization.

Understand:

* learning rate
* convergence
* divergence
* initialization
* number of iterations
* gradient magnitude

### Experiment

Run gradient descent using different learning rates.

For example:

```text
0.001
0.01
0.1
0.5
1.0
```

Plot:

* iteration vs loss
* learning rate vs convergence behavior

### LaTeX

Write the mathematical derivation.

```text
02-mathematics/
└── optimization/
    └── latex/
        └── gradient-descent.tex
```

---

## November 4 — Random Variables & Distributions

### Study

Focus on:

* discrete random variables
* continuous random variables
* probability distributions
* probability mass functions
* probability density functions
* expectation
* variance

### Practice

Use NumPy to simulate distributions.

At minimum:

* Bernoulli
* Binomial
* Normal

---

## November 5 — Statistics

### Study

Focus on:

* mean
* median
* variance
* standard deviation
* covariance
* correlation

Understand the difference between:

$$
Cov(X,Y)
$$

and

$$
Corr(X,Y)
$$

### Implementation

Calculate these quantities manually with NumPy before using built-in functions.

---

## November 6 — Generalization & Overfitting

### Study

**Hands-On Machine Learning — Chapter 2**

Focus on:

* training data
* test data
* generalization
* overfitting
* underfitting
* model complexity

Connect this to October's work.

### Experiment

Train models with increasing complexity and observe:

* training error
* validation error
* test error

---

## November 7 — ML Workflow & Optimization

### Study

Review:

* probability
* statistics
* optimization
* generalization

Read:

* HOML Chapter 2
* ISLR Chapter 2

### Practical work

Construct a complete basic ML workflow:

```text
Dataset
   ↓
Exploration
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Model
   ↓
Training
   ↓
Evaluation
   ↓
Interpretation
```

---

# 5. Week 2 — Classification

## November 8–14

### Goal

Understand how machine-learning models predict discrete classes.

Algorithms:

* Logistic Regression
* K-Nearest Neighbors
* Decision Trees

---

## November 8 — Classification

### Study

**HOML Chapter 3**

Understand:

* binary classification
* multiclass classification
* decision boundaries
* class probabilities
* classification metrics

### Project

Begin working with MNIST or another classification dataset.

---

## November 9 — Logistic Regression

### Study

Understand:

* sigmoid function
* logits
* probability
* decision threshold
* binary classification

Sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Understand why logistic regression can produce probabilities.

---

## November 10 — Logistic Regression From Scratch

Implement logistic regression using NumPy.

Your implementation should include:

* initialization
* forward pass
* sigmoid
* loss
* gradient calculation
* gradient descent
* prediction

Do not use scikit-learn for the core implementation.

---

## November 11 — K-Nearest Neighbors

### Study

Understand:

* distance
* nearest neighbors
* choice of \(k\)
* classification by majority vote
* feature scaling

### Implementation

Implement KNN from scratch.

Support:

```text
fit()
predict()
```

using NumPy.

---

## November 12 — Decision Trees

### Study

**HOML Chapter 6**

Understand:

* nodes
* branches
* leaves
* splitting
* decision boundaries
* recursive partitioning

---

## November 13 — Entropy & Gini Impurity

Study the mathematics behind tree splitting.

Entropy:

$$
H(S)=-\sum_i p_i\log_2(p_i)
$$

Gini impurity:

$$
G(S)=1-\sum_i p_i^2
$$

### LaTeX

Document:

* entropy
* information gain
* Gini impurity

---

## November 14 — Compare Classifiers

Compare:

* Logistic Regression
* KNN
* Decision Tree

Use the same dataset where possible.

Measure:

* accuracy
* precision
* recall
* F1-score
* training time
* prediction behavior

The objective is not to declare one universally superior.

The objective is to understand **why their behavior differs**.

---

# 6. Week 3 — Evaluation, Validation & Feature Engineering

## November 15–21

### Goal

Learn how to determine whether a model actually generalizes.

---

## November 15 — Classification Metrics

Study:

* confusion matrix
* accuracy
* precision
* recall
* F1-score
* false positives
* false negatives

Understand when accuracy can be misleading.

---

## November 16 — Regression Metrics

Study:

### MAE

$$
MAE=\frac{1}{n}\sum_i|y_i-\hat y_i|
$$

### MSE

$$
MSE=\frac{1}{n}\sum_i(y_i-\hat y_i)^2
$$

### RMSE

$$
RMSE=\sqrt{MSE}
$$

### \(R^2\)

Understand what each metric measures and when it is useful.

---

## November 17 — Train / Validation / Test

Understand:

```text
Training set
Validation set
Test set
```

Learn why repeatedly evaluating on the test set can cause problems.

Understand the purpose of a validation set.

---

## November 18 — Cross-Validation

Study:

* k-fold cross-validation
* stratified k-fold
* validation error
* model selection

Implement cross-validation using scikit-learn.

Then understand what the library is doing internally.

---

## November 19 — Feature Engineering

Study:

* categorical variables
* numerical variables
* scaling
* normalization
* standardization
* missing values
* transformations

Understand why feature representation affects model performance.

---

## November 20 — Pipelines & Preprocessing

Learn:

```text
Raw data
   ↓
Preprocessing
   ↓
Feature transformation
   ↓
Model
```

Use:

* `Pipeline`
* `ColumnTransformer`
* `StandardScaler`
* `OneHotEncoder`

Understand why pipelines help prevent data leakage.

---

## November 21 — Model Evaluation Experiment

Perform a complete experiment comparing multiple models.

Record:

* dataset
* preprocessing
* models
* hyperparameters
* validation strategy
* metrics
* results
* observations
* limitations

Create plots showing model performance.

Document the experiment in Markdown.

---

# 7. Week 4 — Ensembles, Clustering & PCA

## November 22–28

### Goal

Understand how multiple models can be combined and how dimensionality can be reduced.

---

## November 22 — Ensemble Learning

### Study

**HOML Chapter 7**

Understand:

* ensemble learning
* voting
* hard voting
* soft voting
* bagging
* boosting

---

## November 23 — Random Forests

### Study

Understand:

* decision-tree ensembles
* bootstrap sampling
* feature randomness
* aggregation
* variance reduction

Read:

**Breiman — Random Forests (2001)**

### Research notes

Create:

```text
07-research/
├── papers/
│   └── random-forests/
└── paper-notes/
    └── random-forests.md
```

Your paper notes should contain:

1. Problem
2. Motivation
3. Main idea
4. Method
5. Important equations
6. Experiments
7. Results
8. Limitations
9. What I learned
10. Questions

---

## November 24 — Gradient Boosting

Understand:

* boosting
* weak learners
* sequential learning
* residual errors
* gradient boosting

Understand the conceptual difference between:

```text
Bagging
```

and

```text
Boosting
```

---

## November 25 — K-Means

### Study

**HOML Chapter 8**

Understand:

* unsupervised learning
* clustering
* centroids
* distance
* cluster assignment
* iterative optimization

K-Means objective:

$$
\min_{\mu_1,\dots,\mu_k}
\sum_{i=1}^{n}
\|x_i-\mu_{c_i}\|^2
$$

---

## November 26 — K-Means From Scratch

Implement K-Means with NumPy.

Your implementation should include:

1. Random centroid initialization
2. Assignment step
3. Centroid update
4. Convergence criterion
5. Final cluster assignments

Experiment with different values of \(k\).

---

## November 27 — PCA Mathematics

Study:

* covariance matrices
* eigenvectors
* eigenvalues
* projections
* dimensionality reduction

Understand the basic mathematical idea behind PCA.

---

## November 28 — PCA With NumPy

Implement PCA without using scikit-learn's PCA implementation.

Your implementation should demonstrate:

```text
Center data
     ↓
Compute covariance matrix
     ↓
Find eigenvalues/eigenvectors
     ↓
Select principal components
     ↓
Project data
```

Compare your result against scikit-learn.

---

# 8. November 29–30 — End-to-End ML Project

## November 29 — Build the Project

Choose a dataset that allows you to demonstrate multiple concepts from the month.

Possible project structure:

```text
06-projects/
└── 03-end-to-end-ml/
    ├── README.md
    ├── data/
    ├── notebooks/
    ├── src/
    ├── models/
    ├── figures/
    └── results/
```

The project should include:

### 1. Problem definition

What are you trying to predict?

### 2. Dataset

Explain:

* source
* features
* target
* number of observations
* missing values
* data types

### 3. Exploratory data analysis

Investigate:

* distributions
* correlations
* outliers
* class balance

### 4. Preprocessing

Document:

* missing-value handling
* scaling
* encoding
* feature engineering

### 5. Baseline model

Build a simple baseline first.

### 6. Multiple models

Possible models:

* Linear Regression
* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* Gradient Boosting

### 7. Evaluation

Use appropriate metrics.

### 8. Error analysis

Investigate where the model performs poorly.

### 9. Experimentation

Change selected:

* features
* hyperparameters
* preprocessing methods
* models

Record the results.

---

# 9. November 30 — Technical Report

Write a technical report documenting the project.

Recommended structure:

```text
1. Introduction

2. Problem Definition

3. Dataset

4. Exploratory Data Analysis

5. Mathematical Background

6. Data Preprocessing

7. Models

8. Experimental Setup

9. Results

10. Error Analysis

11. Discussion

12. Limitations

13. Conclusion

14. Future Work
```

Include:

* equations
* tables
* plots
* experimental results
* references

Use LaTeX where appropriate.

---

# 10. Mathematical Components

November mathematics should focus on understanding the machinery underneath classical ML.

## Probability

You should understand:

* probability
* conditional probability
* independence
* random variables
* distributions
* expectation
* variance

---

## Statistics

You should understand:

* mean
* variance
* covariance
* correlation
* sampling
* estimation
* generalization

---

## Optimization

You should understand:

* objective functions
* gradients
* gradient descent
* learning rates
* convergence
* local minima
* optimization behavior

---

## Classification

You should understand:

* sigmoid
* probability
* decision boundaries
* loss functions
* entropy
* information gain
* Gini impurity

---

## Dimensionality Reduction

You should understand:

* covariance matrices
* eigenvalues
* eigenvectors
* projections
* principal components

---

# 11. Implementations From Scratch

By the end of November, your repository should contain simplified implementations of:

```text
Linear Regression
        ↓
Gradient Descent
        ↓
Logistic Regression
        ↓
KNN
        ↓
Decision Tree
        ↓
K-Means
        ↓
PCA
```

The implementations should primarily use:

* Python
* NumPy

Avoid hiding the important mathematics behind high-level libraries.

For example, your logistic regression implementation should expose the mechanics of:

$$
z=Xw+b
$$

$$
\hat y=\sigma(z)
$$

$$
L(y,\hat y)
$$

$$
\nabla_w L
$$

rather than simply calling:

```python
LogisticRegression()
```

The scikit-learn implementation should then be used as a reference and practical tool.

---

# 12. Experimentation Standard

Every significant algorithm should have at least one experiment.

For example:

### Gradient Descent

Experiment with:

* learning rates
* initialization
* iterations

### KNN

Experiment with:

* different \(k\)
* feature scaling
* distance metrics

### Decision Trees

Experiment with:

* tree depth
* minimum samples
* overfitting

### Random Forests

Experiment with:

* number of trees
* maximum depth
* feature selection

### K-Means

Experiment with:

* number of clusters
* initialization
* convergence

### PCA

Experiment with:

* number of components
* explained variance

---

# 13. LaTeX Standard

LaTeX becomes increasingly important during November.

Do not merely write equations.

Use LaTeX to demonstrate that you understand the derivation.

For example:

```text
02-mathematics/
└── optimization/
    └── latex/
        └── gradient-descent.tex
```

A mathematical note should ideally contain:

```text
Problem
   ↓
Definitions
   ↓
Objective function
   ↓
Derivation
   ↓
Algorithm
   ↓
Interpretation
   ↓
Experiment
```

Important November LaTeX topics:

* probability notation
* expectation
* variance
* covariance
* gradients
* loss functions
* entropy
* Gini impurity
* optimization
* eigenvalues
* eigenvectors
* PCA

---

# 14. Research Paper Reading

November introduces serious paper reading.

Start with:

## Paper 1

**Breiman — Random Forests (2001)**

Purpose:

Understand how an important classical ML method was formally introduced and evaluated.

---

## Paper 2

A foundational treatment of PCA/dimensionality reduction.

Focus on understanding:

* the problem
* mathematical formulation
* principal components
* variance maximization
* projection

Do not worry about understanding every historical detail.

---

## Paper 3

A foundational work on gradient boosting.

Focus on:

* boosting
* weak learners
* sequential improvement
* loss minimization

---

# 15. Research Paper Reading Method

For every paper, answer:

### Problem

What problem is the paper addressing?

### Motivation

Why does the problem matter?

### Existing approaches

What existed before the proposed method?

### Method

What is the proposed approach?

### Mathematics

What are the important equations?

### Experiments

How did the authors test the method?

### Results

What did they observe?

### Limitations

What assumptions or weaknesses exist?

### My understanding

Explain the method in your own words.

### Questions

What remains unclear?

---

# 16. Daily Learning Workflow

Each study day should follow:

```text
1. Read
      ↓
2. Understand
      ↓
3. Derive
      ↓
4. Implement
      ↓
5. Experiment
      ↓
6. Explain
      ↓
7. Document
```

Not every topic requires all seven steps equally.

For example:

### A mathematical concept

Prioritize:

```text
Understand → Derive → Implement → Explain
```

### An ML algorithm

Prioritize:

```text
Understand → Derive → Implement → Experiment → Explain
```

### A research paper

Prioritize:

```text
Read → Understand → Reconstruct → Explain → Question
```

---

# 17. Daily Research Log

From November onward, maintain daily research logs.

Recommended structure:

```text
00-roadmap/
└── daily-log/
    ├── day-032.md
    ├── day-033.md
    ├── ...
    └── day-061.md
```

Each log should contain:

```markdown
# Day XX — Topic

## Topics

## Resources

## What I Learned

## Mathematics

## Implementation

## Experiment

## Results

## Questions

## What I Still Don't Understand

## Evidence of Understanding

## Next Step
```

The important section is:

> **What I Still Don't Understand**

Research ability requires being able to identify gaps in your own understanding.

---

# 18. November GitHub Deliverables

By November 30, the repository should contain evidence of:

## Mathematics

* Probability notes
* Statistics notes
* Optimization notes
* Gradient descent derivation
* Entropy derivation
* Gini impurity derivation
* PCA mathematics

## Implementations

* Linear Regression
* Gradient Descent
* Logistic Regression
* KNN
* Decision Tree
* K-Means
* PCA

## Experiments

* Learning-rate experiments
* Classification comparison
* Model evaluation experiment
* Random Forest experiment
* K-Means experiment
* PCA experiment

## Research

* Random Forests paper notes
* PCA paper/theory notes
* Gradient boosting paper notes

## Project

* One complete end-to-end ML project
* Technical report
* Figures
* Results
* Reproducible code

---

# 19. November Success Criteria

By the end of November, you should be able to:

### Mathematics

* Explain probability and conditional probability.
* Explain expectation and variance.
* Calculate covariance and correlation.
* Explain optimization.
* Derive basic gradient-descent updates.
* Explain entropy and Gini impurity.
* Explain eigenvalues and eigenvectors at a practical level.
* Explain the mathematical intuition behind PCA.

### Machine Learning

* Explain supervised vs unsupervised learning.
* Explain regression vs classification.
* Explain logistic regression.
* Explain KNN.
* Explain decision trees.
* Explain random forests.
* Explain gradient boosting.
* Explain K-Means.
* Explain PCA.

### Implementation

* Implement gradient descent using NumPy.
* Implement logistic regression from scratch.
* Implement KNN from scratch.
* Implement a simplified decision tree.
* Implement K-Means from scratch.
* Implement PCA from scratch.
* Use scikit-learn correctly.

### Evaluation

* Use train/validation/test splits.
* Explain cross-validation.
* Select appropriate metrics.
* Interpret confusion matrices.
* Diagnose overfitting.
* Detect potential data leakage.
* Compare models experimentally.

### Research

* Read a classical ML paper.
* Extract its central contribution.
* Understand its experimental methodology.
* Reproduce a simplified experiment.
* Document questions and limitations.
* Write mathematical explanations using LaTeX.

---

# 20. November Deliverable Checklist

At the end of the month, verify that you have:

* [ ] Probability notes
* [ ] Statistics notes
* [ ] Optimization notes
* [ ] Gradient descent implementation
* [ ] Gradient descent LaTeX derivation
* [ ] Logistic regression implementation
* [ ] KNN implementation
* [ ] Decision-tree implementation
* [ ] K-Means implementation
* [ ] PCA implementation
* [ ] Classification comparison experiment
* [ ] Model evaluation experiment
* [ ] Random Forest experiment
* [ ] K-Means experiment
* [ ] PCA experiment
* [ ] Random Forest paper notes
* [ ] Gradient boosting paper notes
* [ ] PCA research notes
* [ ] End-to-end ML project
* [ ] Technical project report
* [ ] Daily research logs
* [ ] Clean GitHub documentation

---

# 21. Repository Structure After November

Your repository should now be growing toward:

```text
ml-research-sprint-2026/
│
├── 00-roadmap/
│   ├── october.md
│   ├── november.md
│   ├── december.md
│   └── daily-log/
│
├── 01-foundations/
│   ├── python/
│   ├── numpy/
│   ├── pandas/
│   └── visualization/
│
├── 02-mathematics/
│   ├── linear-algebra/
│   ├── calculus/
│   ├── probability/
│   ├── statistics/
│   └── optimization/
│
├── 03-classical-ml/
│   ├── regression/
│   ├── classification/
│   ├── ensemble/
│   ├── clustering/
│   └── dimensionality-reduction/
│
├── 04-implementations-from-scratch/
│   ├── linear-regression/
│   ├── gradient-descent/
│   ├── logistic-regression/
│   ├── knn/
│   ├── decision-tree/
│   ├── k-means/
│   └── pca/
│
├── 05-deep-learning/
│
├── 06-projects/
│   ├── 01-python-data-project/
│   ├── 02-classical-ml/
│   └── 03-end-to-end-ml/
│
├── 07-research/
│   ├── papers/
│   ├── paper-notes/
│   ├── reproductions/
│   ├── experiments/
│   └── research-ideas/
│
├── 08-resources/
│
└── 09-final-report/
```

Do not create empty files merely to fill this structure.

Create directories and files as the corresponding work actually begins.

---

# 22. Guiding Principle

November is where you transition from:

> **"I know how to use machine-learning libraries."**

toward:

> **"I understand what the algorithms are doing, why they work, how to implement them, and how to test whether they actually work."**

The standard is not the number of algorithms completed.

The standard is **depth of understanding**.

For every important algorithm, aim to reach:

$$
\boxed{
\text{Intuition}
\rightarrow
\text{Mathematics}
\rightarrow
\text{Implementation}
\rightarrow
\text{Experiment}
\rightarrow
\text{Explanation}
}
$$

That progression is the bridge from learning machine learning to eventually doing machine-learning research.
