#  Breast Cancer Classification & Data Analysis

## Overview

This project explores the Breast Cancer Wisconsin dataset using exploratory data analysis, dimensionality reduction (PCA), and machine learning classification.

The goal is to understand how cellular features can be used to distinguish between malignant and benign tumors.

---

## Objectives

- Perform exploratory data analysis on biomedical data  
- Visualize relationships between features  
- Reduce dimensionality using PCA  
- Train a machine learning model for classification  
- Evaluate model performance  

---

## Dataset

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, available through `scikit-learn`.

It contains:
- 30 numerical features describing characteristics of cell nuclei
- A binary classification target:
  - 0 → malignant
  - 1 → benign

---

## Methods

### 1. Data Exploration
- Loaded dataset using `sklearn.datasets`
- Converted data into a Pandas DataFrame
- Checked class distribution

### 2. Data Visualization
- Correlation heatmap to explore relationships between features

### 3. Dimensionality Reduction
- Applied Principal Component Analysis (PCA)
- Reduced 30 features to 2 principal components for visualization

### 4. Machine Learning
- Split dataset into training and test sets (80/20)
- Trained a Random Forest Classifier
- Evaluated model using accuracy score

---

## Results

The analysis demonstrates how machine learning models can effectively classify biomedical data using simple statistical and computational techniques.
- PCA visualization shows partial separation between classes  
- The Random Forest model achieved high accuracy on the test set  

*(Results may vary slightly depending on random state)*

---

## Technologies Used

- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Project Structure

- analysis.py
- README.md
- plots /
    - correlation_heatmap.png
    - pca_visualization.png

## Future Improvements

- Test additional models (SVM, Logistic Regression)  
- Perform feature selection analysis  
- Apply cross-validation for more robust evaluation  
- Hyperparameter tuning for model optimization  
- Extend analysis to real gene expression datasets  

---

## Author

Alicia Mañas  
Bioinformatics Student (2nd year)  
Barcelona, Spain