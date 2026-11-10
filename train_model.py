"""
train_model.py
----------------
This script trains a Machine Learning model that reads social media
captions and learns to predict the CATEGORY of the post
(Travel, Food, Fashion, Technology, Fitness, Photography,
Education, Entertainment, Nature, Sports).

Steps performed here:
1. Load the dataset (dataset.csv)
2. Clean/preprocess the caption text
3. Convert text into numbers using TF-IDF
4. Split data into training and testing sets
5. Train a Random Forest Classifier
6. Check accuracy on the test set
7. Save the trained model and the TF-IDF vectorizer to disk (.pkl files)

Run this file ONCE before running the Streamlit app:
    python train_model.py
"""

import re
import string
import warnings

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hide unnecessary warnings so the terminal output stays clean
warnings.filterwarnings("ignore")


def clean_text(text):
    """
    Basic text preprocessing:
    - Convert to lowercase
    - Remove emojis and non-ASCII characters
    - Remove punctuation
    - Remove extra spaces
    This makes sure captions like "Exploring MOUNTAINS!! 🏔️😍" and
    "exploring mountains" are treated the same way by the model.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()                                   # lowercase everything
    text = text.encode("ascii", "ignore").decode("ascii")  # remove emojis / non-ascii
    text = re.sub(r"http\S+|www\S+", " ", text)             # remove links, if any
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                # remove extra spaces

    return text


def main():
    print("Step 1: Loading dataset...")
    df = pd.read_csv("dataset.csv")

    # Clean the caption column
    print("Step 2: Cleaning caption text...")
    df["clean_caption"] = df["caption"].apply(clean_text)

    # Features (X) = cleaned captions, Labels (y) = category
    X = df["clean_caption"]
    y = df["category"]

    print("Step 3: Converting text into numbers using TF-IDF...")
    # TF-IDF turns text into a matrix of numbers based on word importance.
    # max_features limits vocabulary size so the model stays simple & fast.
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X_vectors = vectorizer.fit_transform(X)

    print("Step 4: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Step 5: Training the Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Step 6: Evaluating the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    print("Step 7: Saving model and vectorizer to disk...")
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("\nTraining complete!")
    print("Files created: model.pkl, vectorizer.pkl")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
