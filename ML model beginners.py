# ─────────────────────────────────────────────────────────────────
# BLOCK 1 — Load our tools
# Think of this like opening apps on your phone before you use them.
# Each library here does one job, listed in the comments.
# ─────────────────────────────────────────────────────────────────

# First time running this in Colab? Uncomment the line below to install
# the Hugging Face datasets library, then re-run this cell:
# !pip install datasets -q

import pandas as pd
from datasets import load_dataset                              # pulls public datasets from Hugging Face
from sklearn.feature_extraction.text import TfidfVectorizer    # turns text into numbers
from sklearn.linear_model import LogisticRegression            # our ML model
from sklearn.model_selection import train_test_split           # splits data into train + test
from sklearn.metrics import accuracy_score, classification_report  # measures how well the model did


# ─────────────────────────────────────────────────────────────────
# BLOCK 2 — Load the data
# AG News is a public dataset of 128,000 news articles, each
# tagged with one of 4 categories. We'll teach our model to
# predict the category from the text alone.
# Dataset page: https://huggingface.co/datasets/fancyzhx/ag_news
# ─────────────────────────────────────────────────────────────────

# One line pulls the whole dataset from Hugging Face — no auth, no download
dataset = load_dataset('fancyzhx/ag_news')

# Convert it to a pandas DataFrame so it's easy to look at and filter
df = pd.DataFrame(dataset['train'])

# The dataset stores labels as numbers (0–3). We'll add a readable
# column so we can actually see what each number means.
label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Tech'}
df['label_name'] = df['label'].map(label_names)

# Use a 20,000-article sample so training finishes in seconds.
# Feel free to bump this up later if you want to push accuracy higher.
df = df.sample(20000, random_state=42).reset_index(drop=True)

# Peek at 5 random articles to see what we're working with
df[['text', 'label_name']].head()


# ─────────────────────────────────────────────────────────────────
# BLOCK 3 — Prepare the data
# The model can't read words, only numbers. So we'll convert each
# article into a list of numbers that captures which words appear
# and how distinctive they are. This is called TF-IDF (Term Frequency-Inverse Document Frequency).
# ─────────────────────────────────────────────────────────────────

X = df['text']    # the inputs — what the model learns from
y = df['label']   # the labels — what we want it to predict
                  # (0=World, 1=Sports, 2=Business, 3=Tech)

# TF-IDF: common words like "the" get a low score, rare words like
# "merger" or "goalkeeper" get a high score. The high-score words
# are usually the ones that tell us what category the article is.
vectorizer = TfidfVectorizer(
    max_features=5000,        # ← TRY CHANGING THIS! Lower = less to learn from
    stop_words='english',     # skip common filler words ("the", "and", "is"...)
    ngram_range=(1, 2)        # also learn 2-word phrases like "stock market"
)
X_vectors = vectorizer.fit_transform(X)

# Hold back 20% of the data as a test set the model has never seen.
# This is how we'll honestly measure if it actually learned.
# stratify=y keeps the 4 categories balanced across train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42, stratify=y
)


# ─────────────────────────────────────────────────────────────────
# BLOCK 4 — Train the model
# These two lines are what machine learning actually looks like.
# The model reads 16,000 articles and figures out, on its own,
# which words signal which category.
# ─────────────────────────────────────────────────────────────────

# C is a regularization knob: lower = more cautious, higher = more aggressive
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)  # ← TRY CHANGING C!

# .fit() is where the learning happens. Same method works for fraud
# detection, churn models, medical imaging — different data, same API.
model.fit(X_train, y_train)


# ─────────────────────────────────────────────────────────────────
# BLOCK 5 — Evaluate the model
# Now we test it on articles it has never seen.
# A random guess across 4 categories would be 25% accurate.
# State-of-the-art models hit ~95%. Let's see where we land.
# ─────────────────────────────────────────────────────────────────

predictions = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, predictions):.1%}')
print()

# The classification report breaks accuracy down by category — useful
# for spotting which categories the model confuses (e.g. Business vs Tech)
print(classification_report(
    y_test, predictions,
    target_names=['World', 'Sports', 'Business', 'Tech']
))


# ─────────────────────────────────────────────────────────────────
# BLOCK 6 — Try it on a brand new headline
# Change the headline below to anything you want and re-run this cell.
# What does the model predict? When does it get things wrong?
# ─────────────────────────────────────────────────────────────────

new_headline = ['']

# Run the headline through the same TF-IDF vectorizer we trained earlier,
# then ask the model to predict
new_vector = vectorizer.transform(new_headline)
prediction = model.predict(new_vector)[0]

print('Prediction:', label_names[prediction])