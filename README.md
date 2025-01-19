# X-tweet-sentiment-analysis

This project classifies the sentiment of Twitter posts into categories such as positive, negative, or neutral.

## Libraries Used
- **NumPy & Pandas**: For data manipulation and analysis.
- **NLTK**: For natural language processing tasks like tokenization, stopword removal, and stemming.
- **Scikit-learn**: For machine learning tasks, including model training and evaluation.

## Overview
The goal of this sentiment analysis project is to process a dataset of tweets and use machine learning models to predict whether the sentiment of each tweet is positive, negative, or neutral. The process includes data cleaning, text preprocessing, feature extraction, model training, and evaluation.

### 1. Data Preprocessing:
- We clean the data by removing unnecessary symbols, punctuation, and emojis from the tweet text.
- Stopwords (common words that do not contribute much meaning) are removed to reduce noise in the text.
- Stemming is used to reduce words to their root forms (e.g., "running" becomes "run").

### 2. Feature Extraction:
- The **TF-IDF** (Term Frequency-Inverse Document Frequency) method is used to convert the text data into numerical features, representing the importance of words in the text.

### 3. Model Building:
- A **RandomForestClassifier** is used for classification. It's an ensemble learning method that works well for this type of task by combining the results of multiple decision trees.

### 4. Evaluation:
- The model's performance is measured using **accuracy score**, comparing the predicted sentiment against the actual sentiment labels.
