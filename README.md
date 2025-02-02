# X-tweet-sentiment-analysis
## Overview
The goal of this project is to process a dataset of tweets and use machine learning models to predict whether the sentiment of each tweet is positive, negative, or neutral. The process includes data cleaning, text preprocessing, feature extraction, model training, and evaluation.
## Libraries Used
- **NumPy & Pandas**
- **NLTK**
- **Scikit-learn**
## Data Source:
- The data chosen is available on kaggle.
- https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis 
### 1. Data Preprocessing:
- We clean the data by removing unnecessary symbols, punctuation, and emojis from the tweet text.
- Stopwords (common words that do not contribute much meaning) are removed to reduce noise in the text.
- Stemming is used to reduce words to their root forms (e.g., "running" becomes "run").

### 2. Feature Extraction:
- The **TF-IDF** (Term Frequency-Inverse Document Frequency) method is used to convert the text data into numerical features, representing the importance of words in the text.

### 3. Model Building:
- A **RandomForestClassifier** is used for classification. It's an ensemble learning method that works well for this type of task by combining the results of multiple decision trees.

### 4. Evaluation:
- The model's performance was evaluated using accuracy score.
- Train data accuracy: 95%
- Test data accuracy: 88%

## Clone the Repo
To clone this repository to your local machine, run the following command in your terminal or command prompt:

```bash

git clone https://github.com/gitmonk180/X-tweet-sentiment-analysis.git
