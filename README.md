# ML Journey 🤖

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A structured repository documenting my journey through Machine Learning, featuring comprehensive code examples, detailed notes, and hands-on projects.

## 📚 Repository Structure

The repository is organized into four main sections:

1. Lessons (Files prefixed with `l`)
2. Mathematics (Files prefixed with `m`)
3. Projects (Files prefixed with `p`)
4. Exercises (Files prefixed with `x`)

## Lessons

- [l001_linear_regression.ipynb](l001_linear_regression.ipynb):

  - Linear models ($f_{(\theta)}(x) = \theta^T x$)
  - Mean Squared Error ($J_{(\theta)} = \frac{1}{2m} \sum_{i=1}^{m}(f_{(\theta)}(x^{(i)}) - y^{(i)})^2$)
  - Gradient Descent ($\theta_j := \theta_j - \alpha \frac{\partial J_{(\theta)}}{\partial \theta_j}$)
  - Normal Equation ($\theta = (X^TX)^{-1}X^Ty$)
  - Convergence tests

- [l002_feature_scaling.ipynb](l002_feature_scaling.ipynb):

  - Min-Max Scaling ($X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$)
  - Standardization ($X_{std} = \frac{X - \mu}{\sigma}$)
  - Mean Normalization ($X_{norm} = \frac{X - \mu}{X_{max} - X_{min}}$)

- [l003_logistic_regression.ipynb](l003_logistic_regression.ipynb):

  - Sigmoid function ($\sigma(z) = \frac{1}{1 + e^{-z}}$)
  - Classification decision boundaries
  - Logistic loss function ($-y \log(f(x)) - (1-y) \log(1-f(x))$)
  - Maximum Likelihood Estimation (MLE)
  - Non-linear decision boundaries with polynomial features

- [l004_regularization.ipynb](l004_regularization.ipynb):

  - Underfitting vs. overfitting
  - Bias-variance tradeoff
  - L2 regularization ($J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$)
  - Regularization in gradient descent
  - Impact of regularization parameter ($\lambda$)

- [l005_neural_networks.ipynb](l005_neural_networks.ipynb):

  - Neural network architecture (input, hidden, and output layers)
  - Activation functions (sigmoid, ReLU)
  - Forward propagation
  - TensorFlow implementation
  - Binary classification with neural networks

- [l006_softmax.ipynb](l006_softmax.ipynb):

  - Multiclass classification
  - Softmax function ($a_j = \frac{e^{z_j}}{\sum_{k=1}^{n} e^{z_k}}$)
  - Cross-entropy loss ($L = -\log(a_j)$ for true label $y = j$)
  - Multi-label classification

- [l007_backpropagation.ipynb](l007_backpropagation.ipynb):

  - Computation graphs and forward propagation
  - Chain rule application in neural networks ($\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$)
  - Step-by-step backpropagation algorithm
  - Automatic differentiation (AutoDiff)

- [l008_diagnostics.ipynb](l008_diagnostics.ipynb):

  - Train/Test/Cross-Validation data splitting
  - Model selection and evaluation
  - Bias vs. Variance tradeoff
  - Learning curves
  - Impact of regularization on model performance
  - Evaluating models on imbalanced data (Precision, Recall, F1 score)

- [l009_development_process.ipynb](l009_development_process.ipynb):

  - Iterative ML development workflow
  - Error analysis techniques
  - Data augmentation and synthesis
  - Transfer learning approaches
  - Ethics, fairness, and bias in machine learning

- [l010_decision_trees.ipynb](l010_decision_trees.ipynb):

  - Entropy as a measure of impurity ($H(p_1) = -p_1 \log_2(p_1) - (1-p_1) \log_2(1-p_1)$)
  - Information Gain ($\text{IG} = H(p_1^{\text{root}}) - (w^{\text{left}} \cdot H(p_1^{\text{left}}) + w^{\text{right}} \cdot H(p_1^{\text{right}}))$)
  - Regression trees and variance reduction
  - Ensemble methods (Random Forests and XGBoost)

- [l011_clustering.ipynb](l011_clustering.ipynb):

  - K-means algorithm (assignment and update steps)
  - Distortion function ($J(C, \mu) = \frac{1}{m} \sum_{i=1}^{m} \| x^{(i)} - \mu_{C^{(i)}} \|^2$)
  - Centroid calculation ($\mu_k = \frac{1}{|S_k|} \sum_{x^{(i)} \in S_k} x^{(i)}$)
  - Cluster assignment ($C^{(i)} = \arg\min_{k} \| x^{(i)} - \mu_k \|^2$)
  - Elbow method for determining optimal K

- [l012_anomaly_detection.ipynb](l012_anomaly_detection.ipynb):

  - Gaussian density estimation ($p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$)
  - Parameter estimation ($\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$, $\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} \left(x^{(i)} - \mu\right)^2$)
  - Anomaly detection algorithm ($p(x) < \epsilon$ indicates anomaly)
  - Multivariate Gaussian distribution
  - Feature engineering for anomaly detection

- [l013_collaborative_filtering.ipynb](l013_collaborative_filtering.ipynb):

  - User-item interaction matrix
  - Latent factor models
  - Collaborative filtering cost function ($J(x,\theta) = \frac{1}{2}\sum_{(i,j):r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n} (x_k^{(i)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n} (\theta_k^{(j)})^2$)
  - Mean normalization for handling new users
  - Recommendation systems evaluation metrics

- [l014_content_based_filtering.ipynb](l014_content_based_filtering.ipynb):

  - Feature engineering for users and items
  - Neural network architectures for recommendation
  - Dot product similarity ($\hat{y}^{ij} = v_u^j \cdot v_m^i$)
  - Retrieval and ranking in recommendation systems
  - Ethical considerations in recommender systems

- [l015_pca.ipynb](l015_pca.ipynb):

  - Dimensionality reduction techniques
  - Principal component calculation
  - Explained variance ratio
  - Data projection and reconstruction
  - Applications in data visualization and compression

- [l016_reinforcement_learning.ipynb](l016_reinforcement_learning.ipynb):

  - Markov Decision Processes (MDPs)
  - State-action value function ($Q(s, a) = R(s) + \gamma \max_{a'} Q(s', a')$)
  - Bellman equation
  - Deep Q-Networks (DQN)
  - Exploration vs. exploitation (epsilon-greedy policy)
