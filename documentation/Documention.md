# Project: Fake News Classifier – NLP Challenge (Week 30)

## Project Overview

This project builds a binary classifier to distinguish between **real** (1) and **fake** (0) news articles using only the headline and full text. The solution applies natural language processing and machine learning techniques to achieve near‑perfect accuracy.

## Dataset

* **Source:** Fake News Detection dataset (Kaggle)
* **Size:** \~40,000 labeled news articles
* **Columns:** `label`, `title`, `text`, `subject`, `date`
* **Class balance:** \~50% real, \~50% fake (balanced)

## Methodology

1. **Preprocessing**

   * Lowercasing, removing punctuation/digits
   * Stopword removal
   * POS‑aware lemmatisation (using NLTK)
2. **Feature Engineering**

   * Three feature sets: `title\\\_only`, `text\\\_only`, `combined` (title + text)
3. **Model Selection**

   * Classifiers: Logistic Regression, LinearSVC, Random Forest
   * Hyperparameter tuning via GridSearchCV
   * Evaluation metric: weighted F1 score
   * Cross‑validation: 5‑fold stratified CV on training set
4. **Final Model**

   * Best configuration: Logistic Regression with `combined` feature, `C=10`, `penalty='l1'`
   * Achieved **mean CV F1 = 0.9965** and **test F1 = 1.0000**

## Results

|Metric|Value|
|-|-|
|Accuracy|1.0000|
|Weighted F1 Score|1.0000|
|Confusion Matrix|\[\[5984, 1], \[0, 6000]]|
|Best Feature|combined (title + text)|
|Best Model|LogisticRegression|
|Best Hyperparams|`{'C': 10.0, 'penalty': 'l1', 'solver': 'liblinear'}`|

## How to Reproduce

### Requirements

Install dependencies:

bash:
pip install -r requirements.txt

