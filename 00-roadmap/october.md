# October 2026 — Machine Learning Foundations

> **Sprint:** ML Research Sprint 2026
> **Period:** October 1–31, 2026
> **Theme:** Python → Mathematics → Classical Machine Learning
> **Level:** Beginner
> **Primary objective:** Build the foundations required to study machine learning rigorously.

---

## 1. October Objective

October is the foundation month of the ML Research Sprint.


The goal is to develop the ability to:

* write basic Python confidently
* work with NumPy and pandas
* understand the basic ML workflow
* understand vectors, matrices, and matrix operations
* understand basic calculus needed for ML
* understand probability/statistics fundamentals
* understand linear regression
* understand gradient descent
* understand classification
* implement simple algorithms from scratch
* use scikit-learn appropriately
* document mathematical reasoning using LaTeX

The central learning cycle is:

> **Learn → Understand → Derive → Implement → Experiment → Explain → Review**

---

# 2. Core Resources

## Primary Books

### Python

**Python Crash Course — Eric Matthes**

Primary use:

* Python syntax
* variables
* lists
* loops
* functions
* classes
* dictionaries
* files
* exceptions

---

### Mathematics

**Mathematics for Machine Learning — Deisenroth, Faisal & Ong**

Primary use:

* linear algebra
* calculus
* probability
* mathematical foundations of ML

---

### Machine Learning

**Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurélien Géron**

Primary use:

* ML concepts
* ML workflow
* regression
* classification
* model evaluation
* practical implementation

---

### Linear Algebra Reference

**Introduction to Applied Linear Algebra — Boyd & Vandenberghe**

Use selectively for:

* vectors
* matrices
* linear maps
* inner products
* projections
* linear equations

---

# 3. October Reading Map

| Date       | Python Crash Course | Mathematics for ML | Hands-On ML | Practical Focus                     |
| ---------- | ------------------- | ------------------ | ----------- | ----------------------------------- |
| **Oct 1**  | Ch. 1–2             | Ch. 1              | Ch. 1       | Python + ML overview                |
| **Oct 2**  | Ch. 3               | —                  | —           | Lists                               |
| **Oct 3**  | Ch. 4               | —                  | —           | Loops                               |
| **Oct 4**  | Ch. 5               | —                  | —           | if statements                       |
| **Oct 5**  | Ch. 6               | —                  | —           | Dictionaries                        |
| **Oct 6**  | Ch. 7–8             | —                  | —           | Input, functions                    |
| **Oct 7**  | Review              | Review             | —           | Python mini-project                 |
| **Oct 8**  | —                   | Ch. 2              | —           | Vectors                             |
| **Oct 9**  | —                   | Ch. 2              | —           | Matrices                            |
| **Oct 10** | —                   | Ch. 2              | —           | Matrix operations                   |
| **Oct 11** | —                   | Ch. 3              | —           | Linear maps                         |
| **Oct 12** | —                   | Ch. 3              | —           | Matrix multiplication               |
| **Oct 13** | —                   | Ch. 2–3            | —           | Linear equations                    |
| **Oct 14** | —                   | Ch. 5              | —           | Probability/statistics introduction |
| **Oct 15** | —                   | Review             | Ch. 1       | ML workflow                         |
| **Oct 16** | —                   | —                  | Ch. 2       | End-to-end ML project               |
| **Oct 17** | —                   | Ch. 9              | Ch. 4       | Linear regression                   |
| **Oct 18** | —                   | Ch. 9              | Ch. 4       | Linear regression                   |
| **Oct 19** | —                   | Ch. 5              | Ch. 4       | Gradient descent                    |
| **Oct 20** | —                   | Ch. 5–7            | Ch. 4       | Optimization                        |
| **Oct 21** | —                   | Review             | —           | Linear regression from scratch      |
| **Oct 22** | —                   | Review             | Ch. 4       | sklearn LinearRegression            |
| **Oct 23** | —                   | —                  | Ch. 2       | Train/test/evaluation               |
| **Oct 24** | —                   | —                  | Ch. 3       | Classification                      |
| **Oct 25** | —                   | —                  | Ch. 3       | Logistic regression                 |
| **Oct 26** | —                   | —                  | Ch. 3       | k-NN                                |
| **Oct 27** | —                   | Ch. 5              | Ch. 3       | Logistic regression mathematics     |
| **Oct 28** | —                   | —                  | Ch. 6       | Decision trees                      |
| **Oct 29** | —                   | —                  | Ch. 4       | Regularization/overfitting          |
| **Oct 30** | —                   | —                  | Ch. 1–4     | End-to-end project                  |
| **Oct 31** | —                   | Review             | Review      | Documentation + final project       |

---

# 4. Week 1 — Python Foundations

## October 1–7

### Topics

* Python syntax
* variables
* strings
* numbers
* lists
* loops
* conditionals
* dictionaries
* functions
* basic file handling
* basic object-oriented concepts

### Resources

**Python Crash Course**

* Chapter 1
* Chapter 2
* Chapter 3
* Chapter 4
* Chapter 5
* Chapter 6
* Chapter 7
* Chapter 8

**Mathematics for ML**

* Chapter 1

**Hands-On ML**

* Chapter 1

### Implementation

Create Python exercises covering:

```text
variables
lists
loops
conditionals
dictionaries
functions
```

### Project

Build a small Python project by October 7.

Possible project:

> Student grade analyzer

Input:

* student names
* scores
* courses

Output:

* average
* highest score
* lowest score
* grade classification

---

# 5. Week 2 — Linear Algebra Foundations

## October 8–14

### Mathematics for ML

Study the linear algebra material covering:

* vectors
* vector spaces
* matrices
* matrix operations
* linear maps
* matrix multiplication
* linear equations

### Introduction to Applied Linear Algebra

Use as a supplementary reference for:

* vectors
* matrices
* linear combinations
* linear maps
* matrix multiplication

### Mathematical notebook

Begin writing your formal mathematical notes in LaTeX.

Create:

```text
02-mathematics/
└── linear-algebra/
    └── latex/
        ├── vectors.tex
        ├── matrices.tex
        ├── linear-maps.tex
        └── matrix-multiplication.tex
```

### Implementation

Use NumPy to implement:

* vector addition
* scalar multiplication
* dot product
* matrix addition
* matrix multiplication
* transpose

---

# 6. Week 3 — Understanding How Machines Learn

## October 15–21

### Hands-On ML

Read:

* Chapter 1 — The Machine Learning Landscape
* Chapter 2 — End-to-End Machine Learning Project
* Chapter 4 — Training Models

### Mathematics

Study the mathematics necessary to understand:

* functions
* derivatives
* gradients
* optimization
* loss functions

### Key concepts

Understand:

```text
Data
 ↓
Model
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Parameter update
 ↓
Repeat
```

### Mathematical objective

Understand:

$$
J(\theta)
$$

and:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla J(\theta_t)
$$

### LaTeX

Create:

```text
02-mathematics/
└── optimization/
    └── latex/
        ├── objective-functions.tex
        ├── gradients.tex
        └── gradient-descent.tex
```

---

# 7. October 21 — Linear Regression From Scratch

Implement linear regression using NumPy.

Do not use:

```python
sklearn.linear_model.LinearRegression
```

for the first implementation.

Understand:

$$
\hat{y}=Xw+b
$$

and:

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

Then implement gradient descent.

Structure:

```text
04-implementations-from-scratch/
└── linear-regression/
    ├── linear_regression.py
    ├── gradient_descent.py
    └── README.md
```

---

# 8. Week 4 — Classical Machine Learning

## October 22–28

### Hands-On ML

Study:

* Chapter 4 — Training Models
* Chapter 3 — Classification
* Chapter 6 — Decision Trees

### Algorithms

Learn:

#### Linear Regression

Regression.

#### Logistic Regression

Classification.

#### k-Nearest Neighbors

Classification.

#### Decision Trees

Classification and regression.

---

# 9. Mathematical Components

For each algorithm, identify its mathematical foundation.

### Linear Regression

```text
Vectors
Matrices
Dot products
Least squares
Gradients
Optimization
```

### Logistic Regression

```text
Linear combination
Sigmoid function
Probability
Loss function
Gradient descent
```

### k-NN

```text
Vectors
Distance
Norms
Feature scaling
```

### Decision Trees

```text
Probability
Entropy
Gini impurity
Information gain
```

---

# 10. Implementations From Scratch

By the end of October, implement at least:

```text
04-implementations-from-scratch/
├── linear-regression/
├── gradient-descent/
├── logistic-regression/
└── knn/
```

A simplified decision-tree implementation can follow as an extension.

---

# 11. October 29–31 — Project

Build your first complete ML project.

The project should include:

```text
Data
 ↓
Exploration
 ↓
Preprocessing
 ↓
Training
 ↓
Evaluation
 ↓
Interpretation
```

Suggested structure:

```text
06-projects/
└── 01-classical-ml/
    ├── README.md
    ├── data/
    ├── notebooks/
    ├── src/
    ├── experiments/
    └── results/
```

---

# 12. LaTeX Standard

For every major mathematical concept, maintain two levels of documentation.

### Informal explanation

```text
notes.md
```

Explain the concept in your own words.

### Formal mathematical treatment

```text
derivation.tex
```

Include:

* definitions
* equations
* derivations
* assumptions
* notation
* examples

For example:

```text
linear-regression/
├── README.md
├── mathematics/
│   └── derivation.tex
├── implementation/
│   └── linear_regression.py
└── experiments/
    └── experiment_01.ipynb
```

---

# 13. Daily Learning Workflow

Every study day should follow this general pattern:

### 1. Read

Read the assigned textbook material.

### 2. Understand

Explain the concept without looking at the book.

### 3. Derive

Write the relevant mathematics in LaTeX.

### 4. Implement

Write the concept in Python/NumPy.

### 5. Experiment

Change parameters and observe what happens.

### 6. Explain

Write a short explanation in your notes.

### 7. Document

Commit the useful work to GitHub.

---

# 14. October Success Criteria

By October 31, I should be able to:

### Python

* write functions
* work with lists/dictionaries
* use loops and conditionals
* work with NumPy
* manipulate basic datasets

### Mathematics

* work with vectors
* work with matrices
* perform matrix multiplication
* calculate dot products
* understand basic derivatives
* understand gradients
* understand basic probability/statistics
* understand gradient descent

### Machine Learning

* explain supervised learning
* explain regression vs classification
* implement linear regression
* implement gradient descent
* implement logistic regression
* implement k-NN
* explain decision trees
* understand train/test splitting
* understand overfitting

### Research habits

* write mathematical notes in LaTeX
* read technical documentation
* implement algorithms from scratch
* run controlled experiments
* document results
* explain technical concepts in your own words

---

# 15. October Deliverables

By October 31, the repository should contain evidence of:

```text
✓ Python exercises
✓ NumPy exercises
✓ Linear algebra notes
✓ Calculus notes
✓ Probability/statistics notes
✓ LaTeX mathematical derivations
✓ Linear regression from scratch
✓ Gradient descent from scratch
✓ Logistic regression from scratch
✓ KNN from scratch
✓ Classical ML experiments
✓ One end-to-end ML project
✓ Technical documentation
```

---

# 16. Guiding Principle

The objective of October is not:

> "Finish three books."

The objective is:

> **Build the foundations that allow you to understand machine learning mathematically and implement what you learn.**

The progression is:

```text
Python
   ↓
NumPy
   ↓
Linear Algebra
   ↓
Calculus
   ↓
Probability & Statistics
   ↓
Optimization
   ↓
Linear Regression
   ↓
Classification
   ↓
Classical ML
   ↓
End-to-End Project
```

This forms the foundation for **November's deeper classical ML work** and **December's transition into deep learning and research.**
