# December 2026 — Deep Learning & Research Transition

> **Sprint:** ML Research Sprint 2026
> **Period:** December 1–31, 2026
> **Theme:** Neural Networks → Backpropagation → Optimization → CNNs → Attention → Transformers → Research
> **Level:** Intermediate
> **Primary objective:** Understand the mathematical and computational foundations of modern deep learning and begin developing the habits required for machine-learning research.

---

# 1. December Objective

October established the foundations.

November developed classical machine learning.

December moves into the machinery behind modern AI systems.

The progression is:

```text
Classical ML
     ↓
Neural Networks
     ↓
Forward Propagation
     ↓
Loss Functions
     ↓
Gradients
     ↓
Backpropagation
     ↓
Optimization
     ↓
CNNs
     ↓
Attention
     ↓
Transformers
     ↓
Research Papers
     ↓
Experiments
```

The goal is **not** to become an LLM engineer in one month.

The goal is to understand the computational ideas that eventually lead to systems such as modern large language models.

By December 31, you should be able to explain, implement, experiment with, and mathematically reason about the core components of a neural network.

---

# 2. Core Resources

## Primary resources

### Hands-On Machine Learning

**Aurélien Géron**

Primary chapters:

* Chapter 10 — Introduction to Artificial Neural Networks with Keras
* Chapter 11 — Training Deep Neural Networks

Use HOML for practical implementation and engineering intuition.

---

## Dive into Deep Learning

**Zhang et al.**

Use D2L as the main bridge toward deeper theoretical understanding.

Focus on:

* neural networks
* multilayer perceptrons
* forward propagation
* backpropagation
* optimization
* CNNs
* attention
* Transformers

---

## Deep Learning

**Goodfellow, Bengio & Courville**

Use this as a theoretical reference.

Important chapters:

* Chapter 6 — Deep Feedforward Networks
* Chapter 8 — Optimization for Training Deep Models
* Chapter 9 — Convolutional Networks
* Chapter 10 — Sequence Modeling
* Chapter 11 — Practical Methodology

Do not attempt to read every chapter linearly.

Use the book when you need deeper mathematical explanations.

---

## Mathematics for Machine Learning

Continue using MML as a mathematical reference.

Focus on:

* derivatives
* partial derivatives
* gradients
* chain rule
* optimization
* linear algebra

---

## PyTorch documentation

Use PyTorch for practical deep-learning implementation.

Learn:

* tensors
* datasets
* dataloaders
* modules
* parameters
* autograd
* optimizers
* training loops

---

# 3. December Reading Map

| Date   | Main Resource           | Topic                             |
| ------ | ----------------------- | --------------------------------- |
| Dec 1  | HOML                    | Ch. 10 — Neural Networks          |
| Dec 2  | D2L                     | Neural Networks / MLP             |
| Dec 3  | MML + Deep Learning     | Gradients + Ch. 6                 |
| Dec 4  | D2L                     | Forward propagation               |
| Dec 5  | PyTorch                 | Tensors, datasets, modules        |
| Dec 6  | Deep Learning           | Ch. 6 — Deep Feedforward Networks |
| Dec 7  | Review + implementation | Neural networks                   |
| Dec 8  | MML                     | Calculus/gradients review         |
| Dec 9  | Deep Learning           | Ch. 6 — Backpropagation           |
| Dec 10 | D2L                     | Backpropagation                   |
| Dec 11 | NumPy                   | Neural network from scratch       |
| Dec 12 | NumPy                   | Backpropagation from scratch      |
| Dec 13 | HOML                    | Ch. 11                            |
| Dec 14 | Review + experiments    | Optimization                      |
| Dec 15 | HOML                    | Ch. 10 — MNIST                    |
| Dec 16 | HOML                    | Ch. 11                            |
| Dec 17 | HOML                    | Ch. 11                            |
| Dec 18 | D2L                     | CNNs                              |
| Dec 19 | MML + NumPy             | Convolution                       |
| Dec 20 | D2L + paper             | CNN + LeNet                       |
| Dec 21 | Project                 | MNIST CNN                         |
| Dec 22 | D2L                     | Attention                         |
| Dec 23 | NumPy                   | Self-attention                    |
| Dec 24 | Research paper          | Attention Is All You Need         |
| Dec 25 | Review                  | Transformer architecture          |
| Dec 26 | D2L                     | Transformers                      |
| Dec 27 | PyTorch                 | Training loops                    |
| Dec 28 | PyTorch                 | Mini Transformer                  |
| Dec 29 | Research paper          | AlexNet                           |
| Dec 30 | Experiment              | MLP vs CNN                        |
| Dec 31 | LaTeX                   | Final research report             |

---

# 4. Week 1 — Neural Networks From First Principles

## December 1–7

### Goal

Understand what a neural network actually computes.

You should be able to move from:

$$
x
$$

to:

$$
z=Wx+b
$$

to:

$$
a=f(z)
$$

and understand why stacking these transformations gives a multilayer neural network.

---

# December 1 — Neural Network Foundations

### Study

**HOML Chapter 10**

Understand:

* artificial neurons
* perceptrons
* layers
* inputs
* weights
* biases
* activation functions
* output layers

Understand the basic neuron:

$$
z=w^Tx+b
$$

followed by:

$$
a=f(z)
$$

---

# December 2 — Multilayer Perceptrons

### Study

D2L neural-network material.

Understand:

* input layer
* hidden layers
* output layer
* parameters
* activations
* nonlinearities

Understand why multiple linear layers without nonlinear activation functions do not give you a truly expressive deep network.

---

# December 3 — Mathematical Structure of Neural Networks

### Study

MML calculus material.

Use:

**Deep Learning — Chapter 6**

Focus on:

* functions
* composition
* derivatives
* partial derivatives
* gradients
* chain rule

Represent a simple network mathematically.

For example:

$$
z^{(1)}=W^{(1)}x+b^{(1)}
$$

$$
a^{(1)}=f(z^{(1)})
$$

$$
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}
$$

$$
\hat y=f(z^{(2)})
$$

---

# December 4 — Forward Propagation

Understand forward propagation step by step.

For a simple network:

```text
Input
  ↓
Linear transformation
  ↓
Activation
  ↓
Linear transformation
  ↓
Activation
  ↓
Output
```

Implement forward propagation manually with NumPy.

---

# December 5 — PyTorch Fundamentals

Learn:

* tensors
* tensor shapes
* tensor operations
* datasets
* dataloaders
* neural-network modules
* parameters

Create your first PyTorch model.

Understand what PyTorch is doing rather than treating it as a black box.

---

# December 6 — Deep Feedforward Networks

### Study

**Deep Learning — Chapter 6**

Focus on:

* architecture
* activation functions
* hidden units
* output units
* universal approximation intuition
* computational graphs

Important activations:

* sigmoid
* tanh
* ReLU
* softmax

Understand why ReLU became important in deep learning.

---

# December 7 — Neural Network Implementation

Implement a small neural network using NumPy.

At minimum:

```text
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Output
```

Do not use PyTorch for the core implementation.

Document:

* architecture
* equations
* tensor shapes
* forward pass

---

# 5. Week 2 — Backpropagation & Optimization

## December 8–14

### Goal

Understand the mechanism that allows neural networks to learn.

The conceptual chain is:

```text
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Backpropagation
   ↓
Parameter update
```

---

# December 8 — Calculus Review

Review:

* derivatives
* partial derivatives
* gradients
* chain rule
* Jacobians at a conceptual level

Focus on the chain rule.

For:

$$
y=f(g(x))
$$

we have:

$$
\frac{dy}{dx}
=
\frac{dy}{dg}
\frac{dg}{dx}
$$

This becomes the foundation of backpropagation.

---

# December 9 — Backpropagation

### Study

**Deep Learning — Chapter 6**

Understand:

* computational graphs
* forward pass
* loss
* backward pass
* gradients
* parameter updates

Do not memorize backpropagation.

Understand why the chain rule makes it possible.

---

# December 10 — Backpropagation With D2L

Work through a complete numerical example.

Track:

```text
Input
 ↓
Weights
 ↓
Activation
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Weight update
```

Calculate at least one example manually.

---

# December 11 — Neural Network From Scratch

Build a small neural network using NumPy.

The implementation should include:

* weights
* biases
* forward propagation
* activation functions
* loss
* predictions

---

# December 12 — Backpropagation From Scratch

Extend the network to calculate gradients.

Implement:

```text
forward()
loss()
backward()
update()
```

The goal is to understand what automatic differentiation frameworks automate.

---

# December 13 — Training Deep Networks

### Study

**HOML Chapter 11**

Focus on:

* vanishing gradients
* exploding gradients
* initialization
* normalization
* optimizers
* learning rates

---

# December 14 — Optimization Experiment

Experiment with:

* learning rate
* initialization
* batch size
* optimizer

Compare:

* SGD
* Momentum
* Adam

Understand what each optimizer is doing conceptually.

---

# 6. Week 3 — Practical Deep Learning & CNNs

## December 15–21

### Goal

Train neural networks on real datasets and understand convolutional neural networks.

---

# December 15 — MNIST With Neural Networks

### Study

HOML Chapter 10.

Build a neural network for MNIST.

Understand:

```text
Images
 ↓
Flatten
 ↓
Dense layers
 ↓
Output
 ↓
Class prediction
```

---

# December 16 — Improving Training

Study HOML Chapter 11.

Experiment with:

* learning rate
* batch size
* number of hidden units
* number of layers

Record the results.

---

# December 17 — Regularization

Study:

* dropout
* early stopping
* weight regularization
* initialization
* normalization

Experiment with at least one regularization technique.

---

# December 18 — Convolutional Neural Networks

### Study

D2L CNN material.

Understand:

* convolution
* filters
* kernels
* feature maps
* stride
* padding
* pooling

Conceptually:

```text
Image
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Classifier
```

---

# December 19 — Convolution Mathematics

Implement a simple 2D convolution using NumPy.

Understand:

* kernel
* receptive field
* stride
* padding

Do not rely immediately on a deep-learning library.

---

# December 20 — LeNet

Study:

**LeCun et al. — Gradient-Based Learning Applied to Document Recognition**

Understand:

* convolution
* subsampling/pooling
* feature extraction
* classification

Create paper notes.

```text
07-research/
├── papers/
│   └── lenet/
└── paper-notes/
    └── lenet.md
```

---

# December 21 — MNIST CNN Project

Build a CNN for MNIST using PyTorch.

Record:

* architecture
* number of parameters
* training loss
* validation accuracy
* test accuracy
* training time

Save plots.

Project:

```text
06-projects/
└── 04-mnist-cnn/
    ├── README.md
    ├── src/
    ├── notebooks/
    ├── figures/
    └── results/
```

---

# 7. Week 4 — Attention & Transformers

## December 22–28

### Goal

Understand the fundamental architecture behind modern Transformer-based models.

The progression is:

```text
Sequence Modeling
       ↓
Attention
       ↓
Self-Attention
       ↓
Multi-Head Attention
       ↓
Transformer
```

---

# December 22 — Attention

### Study

D2L attention material.

Understand:

* queries
* keys
* values
* attention weights
* weighted sums

The core idea can be expressed as:

$$
Attention(Q,K,V)
$$

---

# December 23 — Self-Attention From Scratch

Implement scaled dot-product self-attention using NumPy.

The core equation is:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Implement:

1. Query projection
2. Key projection
3. Value projection
4. Score calculation
5. Scaling
6. Softmax
7. Weighted value aggregation

---

# December 24 — Attention Is All You Need

Read:

**Vaswani et al. — Attention Is All You Need**

Focus on:

* motivation
* self-attention
* multi-head attention
* positional encoding
* encoder
* decoder
* residual connections
* layer normalization
* feed-forward networks

Do not attempt to understand every detail on the first pass.

---

# December 25 — Transformer Architecture

Reconstruct the Transformer architecture from memory.

Draw the architecture yourself.

Understand:

```text
Tokens
 ↓
Embeddings
 ↓
Positional Information
 ↓
Self-Attention
 ↓
Feed-Forward Network
 ↓
Residual + Normalization
 ↓
Repeated Layers
```

Understand the difference between:

* encoder
* decoder
* encoder-decoder architecture

---

# December 26 — Transformers With D2L

Study D2L Transformer material.

Focus on:

* attention
* positional encoding
* multi-head attention
* encoder blocks
* decoder blocks

Relate the implementation back to the original paper.

---

# December 27 — PyTorch Training Loops

Build a training loop manually.

Understand:

```python
for X, y in dataloader:

    optimizer.zero_grad()

    predictions = model(X)

    loss = loss_function(predictions, y)

    loss.backward()

    optimizer.step()
```

Know what every line does.

Do not treat the training loop as boilerplate.

---

# December 28 — Mini Transformer

Build a small Transformer model in PyTorch.

The model does not need to be large.

The objective is understanding.

Possible task:

* character-level language modeling
* toy sequence prediction
* simple sequence classification

Document:

* architecture
* parameters
* training procedure
* loss
* results

---

# 8. December 29–30 — Research Experiments

## December 29 — AlexNet

Read:

**Krizhevsky, Sutskever & Hinton — ImageNet Classification with Deep Convolutional Neural Networks**

Understand:

* scale of the dataset
* deep CNN architecture
* ReLU
* dropout
* GPU training
* data augmentation
* ImageNet
* why the work mattered historically

Create:

```text
07-research/
├── papers/
│   └── alexnet/
└── paper-notes/
    └── alexnet.md
```

---

# December 30 — MLP vs CNN

Perform an experiment comparing:

```text
MLP
vs
CNN
```

on an image classification dataset.

Keep the experiment controlled.

Compare:

* architecture
* number of parameters
* training time
* training loss
* validation accuracy
* test accuracy
* generalization

### Research question

> How does incorporating spatial structure through convolution affect image classification compared with a fully connected network?

Document the experiment as a miniature research study.

---

# 9. December 31 — Final Research Report

The final day is dedicated to consolidating the entire three-month sprint.

Create:

```text
09-final-report/
├── main.tex
├── references.bib
├── figures/
└── report.pdf
```

The report should document the intellectual progression of the sprint.

---

# 10. Final Report Structure

## 1. Introduction

Explain:

* why the sprint was undertaken
* objectives
* learning methodology
* scope

---

## 2. Mathematical Foundations

Cover:

* linear algebra
* calculus
* probability
* statistics
* optimization

Include important equations.

---

## 3. Classical Machine Learning

Discuss:

* linear regression
* logistic regression
* KNN
* decision trees
* random forests
* gradient boosting
* K-Means
* PCA

---

## 4. Deep Learning

Discuss:

* neural networks
* forward propagation
* loss functions
* backpropagation
* optimization
* regularization

---

## 5. Convolutional Neural Networks

Explain:

* convolution
* filters
* feature maps
* pooling
* CNN architecture

Discuss the MNIST experiment.

---

## 6. Transformers

Explain:

* attention
* self-attention
* queries
* keys
* values
* positional encoding
* multi-head attention
* Transformer blocks

Include the self-attention equation:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

---

## 7. Implementations

Document important implementations:

* gradient descent
* linear regression
* logistic regression
* KNN
* decision tree
* K-Means
* PCA
* neural network
* backpropagation
* convolution
* self-attention

Explain what was implemented from scratch and what was implemented using frameworks.

---

## 8. Experiments

Document major experiments.

For each experiment:

```text
Research Question
        ↓
Hypothesis
        ↓
Experimental Setup
        ↓
Implementation
        ↓
Results
        ↓
Analysis
        ↓
Limitations
        ↓
Conclusion
```

---

## 9. Research Papers

Discuss the papers studied:

### Random Forests

Breiman, 2001.

### LeNet

LeCun et al.

### AlexNet

Krizhevsky, Sutskever & Hinton.

### Attention Is All You Need

Vaswani et al.

For each paper:

* problem
* method
* contribution
* mathematics
* experiments
* limitations
* what you learned

---

## 10. What I Learned

Do not write generic statements such as:

> "I learned a lot about AI."

Instead discuss specific insights.

Examples:

* why optimization is central to ML
* why representation matters
* why nonlinearities enable deep networks
* why backpropagation scales learning
* why convolution exploits spatial structure
* why attention handles relationships differently
* how experimental design affects conclusions

---

## 11. Open Questions

This section is extremely important.

Document questions that emerged during the sprint.

Examples:

* Why does SGD generalize well despite noisy updates?
* Why does overparameterization sometimes improve generalization?
* Why does attention scale poorly with sequence length?
* What determines useful representations?
* Why do some architectures train more reliably than others?
* How does pretraining produce transferable representations?
* How does scaling affect emergent capabilities?

You are not expected to answer all of these.

The purpose is to begin developing **research questions**.

---

## 12. Future Research Directions

Identify areas you want to investigate during 2027.

Possible areas:

* representation learning
* optimization
* computer vision
* NLP
* Transformers
* efficient inference
* reinforcement learning
* multimodal learning
* interpretability
* AI safety
* scientific machine learning

Do not choose a research specialization simply because it sounds impressive.

Base future directions on what you found technically interesting during the sprint.

---

# 11. December Research Paper Method

Paper reading should now become more rigorous.

For each paper, create:

```text
07-research/
└── paper-notes/
    └── paper-name.md
```

Use:

```markdown
# Paper Title

## Citation

## Problem

## Motivation

## Previous Work

## Main Contribution

## Method

## Mathematical Formulation

## Architecture

## Experimental Setup

## Results

## Limitations

## My Reconstruction

## What I Understand

## What I Don't Understand

## Questions

## Ideas Inspired By This Paper
```

---

# 12. Research Reproduction Standard

By December, do not simply read papers.

Attempt small reproductions.

A reproduction does **not** mean rebuilding the original system at its original scale.

Instead:

> Reproduce the central mechanism or a simplified experiment.

Examples:

### LeNet

Train a small CNN on MNIST.

### AlexNet

Reproduce the conceptual architecture at a much smaller scale.

### Attention

Implement scaled dot-product attention from scratch.

### Transformer

Build a small Transformer on a toy sequence task.

---

# 13. From-Scratch Standard

Your from-scratch implementations should now progress from:

```text
NumPy
```

to:

```text
PyTorch
```

The distinction is important.

### NumPy

Use NumPy when the goal is:

> "I want to understand the mathematics and mechanics."

### PyTorch

Use PyTorch when the goal is:

> "I want to build and train practical deep-learning systems."

You should be comfortable with both.

---

# 14. PyTorch Skills

By the end of December, you should understand:

### Tensors

* shapes
* dimensions
* broadcasting
* matrix multiplication
* device management

### Models

* `nn.Module`
* parameters
* layers
* forward passes

### Training

* loss functions
* optimizers
* gradients
* backpropagation
* training loops

### Data

* datasets
* dataloaders
* batching
* train/validation/test splits

### Experiments

* checkpoints
* metrics
* logging
* plots
* reproducibility

---

# 15. December LaTeX Standard

Your mathematical writing should now become substantially stronger.

Important equations to document:

### Linear layer

$$
z=Wx+b
$$

### Activation

$$
a=f(z)
$$

### Loss

$$
L(y,\hat y)
$$

### Gradient descent

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta J(\theta_t)
$$

### Backpropagation

Use the chain rule to derive gradients through a computational graph.

### Convolution

Document the mathematical operation and indexing.

### Attention

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

LaTeX should communicate the reasoning behind the equation, not merely display it.

---

# 16. December Experiments

Minimum experiments:

## Experiment 1 — Optimization

Compare:

* SGD
* Momentum
* Adam

---

## Experiment 2 — Neural Network Architecture

Compare models with different:

* depths
* widths
* activations

---

## Experiment 3 — Regularization

Compare:

* baseline
* dropout
* weight regularization

---

## Experiment 4 — MLP vs CNN

Compare:

* parameter count
* training behavior
* accuracy
* generalization

---

## Experiment 5 — Attention

Visualize attention weights for a toy sequence.

Investigate which tokens receive attention.

---

# 17. December Project Structure

By the end of December:

```text
05-deep-learning/
├── neural-networks/
├── backpropagation/
├── optimization/
├── pytorch/
├── cnn/
└── transformers/
```

Projects:

```text
06-projects/
├── 01-python-data-project/
├── 02-classical-ml/
├── 03-end-to-end-ml/
├── 04-mnist-cnn/
└── 05-mlp-vs-cnn/
```

Research:

```text
07-research/
├── papers/
├── paper-notes/
├── reproductions/
├── experiments/
└── research-ideas/
```

---

# 18. December Daily Research Log

Continue the same format from October and November.

```text
00-roadmap/
└── daily-log/
    ├── day-062.md
    ├── day-063.md
    ├── ...
    └── day-092.md
```

Each day:

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

The final three sections become increasingly important as you move toward research.

---

# 19. December Success Criteria

By December 31, you should be able to:

## Mathematics

* Explain gradients.
* Apply the chain rule.
* Explain computational graphs.
* Derive simple neural-network gradients.
* Explain optimization for neural networks.
* Explain convolution mathematically.
* Explain the mathematical structure of self-attention.

---

## Neural Networks

* Explain neurons.
* Explain layers.
* Explain weights and biases.
* Explain activation functions.
* Explain forward propagation.
* Explain loss functions.
* Explain backpropagation.
* Explain gradient descent.
* Explain regularization.

---

## Deep Learning

* Build an MLP.
* Train an MNIST classifier.
* Implement a neural network with NumPy.
* Implement backpropagation from scratch.
* Build a CNN.
* Train a CNN with PyTorch.
* Explain why CNNs are useful for images.

---

## Transformers

* Explain attention.
* Explain self-attention.
* Explain queries, keys, and values.
* Explain scaled dot-product attention.
* Explain multi-head attention.
* Explain positional encoding.
* Explain Transformer blocks.
* Read and explain the core ideas of *Attention Is All You Need*.
* Implement a small self-attention mechanism.
* Build a toy Transformer.

---

## Research

You should be able to:

* Read a technical ML paper.
* Identify its research question.
* Identify its contribution.
* Reconstruct its core method.
* Understand important equations.
* Analyze experiments.
* Identify limitations.
* Reproduce a simplified experiment.
* Formulate follow-up questions.
* Document your findings in LaTeX.

---

# 20. December Deliverable Checklist

Before declaring the sprint complete:

### Deep Learning

* [ ] Neural-network notes
* [ ] Forward-propagation implementation
* [ ] Neural network from scratch
* [ ] Backpropagation derivation
* [ ] Backpropagation from scratch
* [ ] Optimization experiments
* [ ] PyTorch training loop
* [ ] MNIST MLP

### Computer Vision

* [ ] Convolution notes
* [ ] NumPy convolution
* [ ] LeNet paper notes
* [ ] MNIST CNN
* [ ] CNN experiment

### Transformers

* [ ] Attention notes
* [ ] Self-attention from scratch
* [ ] Transformer paper notes
* [ ] Transformer architecture diagram
* [ ] Mini Transformer
* [ ] Toy sequence experiment

### Research

* [ ] LeNet paper
* [ ] AlexNet paper
* [ ] Attention Is All You Need
* [ ] Paper notes
* [ ] Simplified reproductions
* [ ] Research questions
* [ ] Experiment documentation

### Final Report

* [ ] `main.tex`
* [ ] `references.bib`
* [ ] figures
* [ ] mathematical derivations
* [ ] experiment results
* [ ] paper analysis
* [ ] open questions
* [ ] future research directions
* [ ] compiled PDF

---

# 21. Final Three-Month Progression

Your complete sprint should now look like:

```text
OCTOBER
────────────────────────────────────
Python
   ↓
NumPy
   ↓
Linear Algebra
   ↓
Calculus
   ↓
Probability
   ↓
Linear Regression
   ↓
Gradient Descent
   ↓
Classification
   ↓
Classical ML


NOVEMBER
────────────────────────────────────
Probability & Statistics
   ↓
Optimization
   ↓
Logistic Regression
   ↓
KNN
   ↓
Decision Trees
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Ensembles
   ↓
K-Means
   ↓
PCA


DECEMBER
────────────────────────────────────
Neural Networks
   ↓
Forward Propagation
   ↓
Backpropagation
   ↓
Optimization
   ↓
CNNs
   ↓
LeNet
   ↓
AlexNet
   ↓
Attention
   ↓
Transformers
   ↓
Research Papers
   ↓
Experiments
   ↓
Final Research Report
```

---

# 22. The December Standard

The biggest change in December is the standard of understanding.

Do not stop at:

> "I can train a neural network."

Aim for:

> "I understand what the network computes, how the loss is constructed, how gradients flow through it, how the parameters are optimized, how the architecture affects learning, how to implement the mechanism myself, and how to design an experiment that tests a meaningful question."

For every major deep-learning concept, aim for:

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
\text{Paper}
\rightarrow
\text{Research Question}
}
$$

That final step is the purpose of the entire October–December sprint.

The objective is not simply to finish three months of material.

The objective is to finish December with the ability to **learn new ML research independently**.
