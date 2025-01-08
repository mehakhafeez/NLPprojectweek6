# Sentiment Analysis and Clustering on Amazon Reviews

This project applies machine learning and deep learning techniques for sentiment analysis and clustering of Amazon reviews. It includes data preprocessing, sentiment labeling using TextBlob, traditional ML models (Logistic Regression and Naive Bayes), clustering with KMeans, and fine-tuning a DistilBERT model for sentiment classification.

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Steps](#steps)
   - [Data Preprocessing](#data-preprocessing)
   - [Sentiment Analysis with TextBlob](#sentiment-analysis-with-textblob)
   - [Logistic Regression and Naive Bayes](#logistic-regression-and-naive-bayes)
   - [KMeans Clustering](#kmeans-clustering)
   - [DistilBERT Fine-Tuning](#distilbert-fine-tuning)
5. [Results](#results)
6. [Visualization](#visualization)
7. [Huggingfacedeployment]
---

## Overview

This project analyzes Amazon product reviews using machine learning and deep learning techniques. It clusters reviews to identify patterns and fine-tunes a pre-trained DistilBERT model for accurate sentiment classification.

---
## Steps

### 1. Data Preprocessing
- Combine the datasets using Pandas.
- Remove duplicates and NaN values from the `reviews.text` column.
- Apply regex to clean non-alphabetic characters from the reviews.

### 2. Sentiment Analysis with TextBlob
- Use TextBlob to calculate polarity scores for each review.
- Assign a sentiment (`positive`, `neutral`, or `negative`) based on the polarity score.
- Add the sentiment labels to a new `sentiment` column in the dataset.

### 3. Logistic Regression and Naive Bayes
- Vectorize the reviews using TF-IDF (5000 features).
- Train Logistic Regression and Naive Bayes classifiers on the vectorized data.
- Evaluate their performance using accuracy metrics.

### 4. KMeans Clustering
- Reduce dimensionality of the TF-IDF matrix using PCA (2 components).
- Apply KMeans clustering to group reviews into 3 clusters.
- Visualize clusters in a 2D scatter plot.

### 5. Fine-Tuning DistilBERT
- Tokenize the reviews using `DistilBertTokenizer`.
- Prepare training and testing datasets for fine-tuning.
- Fine-tune `DistilBertForSequenceClassification` to classify sentiments into 3 classes.
- Evaluate model performance.

### 6. Visualization
- Visualize results of KMeans clustering using a scatter plot of PCA components.
- Optionally, create confusion matrices for sentiment classification models.

### 7. Huggingface deployment
https://huggingface.co/spaces/mehakovais/NLPprojectfinal
https://colab.research.google.com/drive/1mnVNfBr1Hj7Z7bK0HGmKaJjlRiFVwOkb#scrollTo=6wWZHqGNZlOC

## Installation

### Requirements
Ensure Python is installed. Install required packages with:

```bash
pip install -r requirements.txt


