
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
import emoji
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\jaisaikrishna\Downloads\twitter_training.csv\twitter_training.csv")
df = df.iloc[:, 2:]

# Reorganize columns
col_list = list(df.columns)
col_list[0], col_list[-1] = col_list[-1], col_list[0]
df = df[col_list]

# Rename columns
df.rename(columns={df.columns[0]: 'Text', df.columns[1]: 'Sentiment'}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Preprocess text: remove punctuation and stopwords
df['Text'] = df['Text'].str.lower()
def remove_punc(test_str):
    return test_str.translate(str.maketrans('', '', string.punctuation))

df['Text'] = df['Text'].apply(remove_punc)

stop_words = set(stopwords.words('english'))
df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df.dropna(inplace=True)

# Convert emojis to text
def emoji_to_text(text):
    return emoji.demojize(text)

df['Text'] = df['Text'].apply(emoji_to_text)

# Remove 'Irrelevant' sentiment from data as using only 3-pos,neg,neu.
df = df[df['Sentiment'] != 'Irrelevant']


# Remove empty strings
df = df[df['Text'].str.strip().ne('')]

# Stem text( converts taking/taken/took to take)
ps = PorterStemmer()
def stem_text(text):
    if pd.isnull(text):
        return text
    tokens = word_tokenize(str(text))
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['Text'] = df['Text'].apply(stem_text)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text'])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Sentiment'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
train_pred = model.predict(X_train)
ac1 = accuracy_score(train_pred, y_train)
print(f'Training Accuracy: {ac1}')

test_pred = model.predict(X_test)
ac2 = accuracy_score(test_pred, y_test)
print(f'Test Accuracy: {ac2}')

# Testing model with custom text(provided preprocessed texts)
txts = ["wow thats great news", "No dont like it", "new skin got crazy check out"]
ty = vectorizer.transform(txts)
ss = model.predict(ty)
print(f'Sentiment predictions: {ss}')




