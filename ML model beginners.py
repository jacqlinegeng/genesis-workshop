# BLOCK 1
# If running in Colab for the first time, uncomment to install the datasets library:
# !pip install datasets -q

import pandas as pd
from datasets import load_dataset                              # Hugging Face loader
from sklearn.feature_extraction.text import TfidfVectorizer    # text → numbers
from sklearn.linear_model import LogisticRegression            # our ML model
from sklearn.model_selection import train_test_split           # train/test split
from sklearn.metrics import accuracy_score, classification_report


# Load AG News from Hugging Face - the public dataset registry used by every major AI lab
# Dataset page: https://huggingface.co/datasets/fancyzhx/ag_news
dataset = load_dataset('fancyzhx/ag_news')

# Convert the training split to a pandas DataFrame for easy handling
df = pd.DataFrame(dataset['train'])

# Labels in Hugging Face: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
df['label_name'] = df['label'].map(label_names)

# Sample 20,000 articles so the demo runs in seconds
df = df.sample(20000, random_state=42).reset_index(drop=True)

df[['text', 'label_name']].head()   # show the first 5 articles


# BLOCK 2
X = df['text']    # input: the news article text
y = df['label']   # label: 1=World, 2=Sports, 3=Business, 4=Sci/Tech

# Turn text into numbers - top 5000 most informative single words AND word pairs
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)   # also capture 2-word phrases like 'stock market'
)
X_vectors = vectorizer.fit_transform(X)

# 80% for training, 20% for testing (stratified so each class is balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42, stratify=y
)

# BLOCK 3
# Create the model - try changing C later (it's a regularization knob)
model = LogisticRegression(max_iter=1000, random_state=42)

# Train it - this is where the learning actually happens
model.fit(X_train, y_train)


# BLOCK 4
predictions = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, predictions):.1%}')
print()
print(classification_report(
    y_test, predictions,
    target_names=['World', 'Sports', 'Business', 'Sci/Tech']
))


# BLOCK 5
# Type any headline here
new_headline = ['Apple announces breakthrough in quantum computing research']

new_vector = vectorizer.transform(new_headline)
prediction = model.predict(new_vector)[0]

print('Prediction:', label_names[prediction])