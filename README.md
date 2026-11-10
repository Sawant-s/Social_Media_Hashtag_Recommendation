# 📱 Social Media Hashtag Recommendation System

An AIML micro-project that reads a social media caption, predicts its
**category** using Machine Learning, and recommends relevant **hashtags** —
built with Python, Scikit-learn and Streamlit.

---

## 1. Project Structure

```
social_media_hashtag_recommender/
│
├── app.py              → Streamlit web application (the UI)
├── train_model.py       → Script that trains and saves the ML model
├── dataset.csv          → Training data (captions, categories, hashtags)
├── model.pkl             → Saved trained Random Forest model (created after training)
├── vectorizer.pkl       → Saved TF-IDF vectorizer (created after training)
├── requirements.txt      → List of Python libraries needed
└── README.md             → This file (setup guide + viva preparation)
```

---

## 2. Where to Create Each File (VS Code)

1. Open VS Code → **File > Open Folder** → create/select a folder named
   `social_media_hashtag_recommender`.
2. Inside that folder, create the files exactly as named above
   (`app.py`, `train_model.py`, `dataset.csv`, `requirements.txt`, `README.md`).
3. `model.pkl` and `vectorizer.pkl` are **not** created manually — they are
   generated automatically the first time you run `train_model.py`.

---

## 3. Installation Steps

Open a terminal inside the project folder in VS Code and run:

```bash
# (Optional but recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install all required libraries
pip install -r requirements.txt
```

---

## 4. Train the Model

Before running the app, you must train the model once. This reads
`dataset.csv`, trains the Random Forest classifier, and saves
`model.pkl` and `vectorizer.pkl`:

```bash
python train_model.py
```

You will see clean step-by-step output ending with the model accuracy
and a confirmation that the `.pkl` files were saved.

---

## 5. Run the Streamlit Application

```bash
streamlit run app.py
```

This opens the app automatically in your browser (usually at
`http://localhost:8501`). Type a caption, click **Recommend Hashtags**,
and see the predicted category + hashtags appear.

### Example

**Input:** "Exploring the beautiful mountains with my friends."

**Output:**
- Predicted Category: **Travel**
- Recommended Hashtags: `#Travel #Mountains #Adventure #Trekking #Wanderlust #TravelPhotography`

---

## 6. How the Project Works (Simple Flow)

1. `dataset.csv` contains ~130 example captions across 10 categories
   (Travel, Food, Fashion, Technology, Fitness, Photography, Education,
   Entertainment, Nature, Sports), each with matching hashtags.
2. `train_model.py` cleans the text, converts it into numbers using
   **TF-IDF**, and trains a **Random Forest Classifier** to predict the
   category from the text.
3. `app.py` loads the saved model + vectorizer, takes a new caption from
   the user, cleans it the same way, predicts its category, and looks up
   hashtags used by that category in the dataset to recommend them.

---

## 7. Viva Preparation

### 1. Project Objective
To build a simple Machine Learning based web application that reads a
social media caption, predicts which category it belongs to, and
automatically suggests relevant hashtags — helping users increase the
reach and engagement of their posts.

### 2. Problem Statement
Manually choosing the right hashtags for a post is time-consuming and
often based on guesswork. Many users don't know which hashtags are
trending or relevant to their content, which reduces their post's reach.
This project solves that by automatically analyzing the caption's
content and suggesting hashtags matched to its category.

### 3. Why This Project is Needed
- Saves time for content creators, students, and small businesses.
- Improves consistency of hashtag usage.
- Demonstrates a real, practical use of core Machine Learning concepts
  (text classification) rather than a purely theoretical exercise.
- Easy to extend to real-world tools like social media schedulers.

### 4. How the System Works
1. User types a caption in the Streamlit interface.
2. The text is cleaned (lowercase, punctuation/emoji removal).
3. TF-IDF converts the cleaned text into numeric features.
4. The trained Random Forest model predicts the most likely category.
5. Hashtags associated with that category (from the dataset) are shown
   to the user, along with a simple confidence chart.

### 5. What is Supervised Learning?
Supervised learning is a type of Machine Learning where the model is
trained on **labeled data** — meaning every input (caption) already has
a known correct output (category). The model learns the relationship
between input and output during training, so it can predict the
category for new, unseen captions.

### 6. What is TF-IDF?
TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It is
a technique to convert text into numbers by measuring how important a
word is:
- **Term Frequency (TF):** how often a word appears in a document.
- **Inverse Document Frequency (IDF):** how rare/common that word is
  across all documents.
Words that appear often in one caption but rarely across all captions
get a higher score, since they are more meaningful for classification
(e.g., "mountains" for Travel vs common words like "the").

### 7. Why Random Forest / Decision Tree is Used
- Easy to understand and explain (no complex math like neural networks).
- Works well with small-to-medium sized text datasets like this one.
- Random Forest builds many small Decision Trees and combines their
  votes, which usually gives more stable and accurate predictions than
  a single Decision Tree.
- Doesn't require GPU or heavy computation — perfect for a diploma-level
  project.

### 8. What is Training and Testing?
- **Training data** is the portion of the dataset (80% here) used to
  teach the model the patterns between captions and categories.
- **Testing data** is the remaining portion (20% here) that the model
  has never seen before. It is used to check how well the model
  performs on new/unseen data, giving an honest measure of accuracy.

### 9. What is the Input and Output?
- **Input:** A social media caption typed by the user (plain text).
- **Output:** 
  - The predicted category of the caption (e.g., Travel, Food).
  - A list of 5–10 recommended hashtags relevant to that category.

### 10. Advantages and Limitations

**Advantages:**
- Simple, fast, and beginner-friendly to build and explain.
- Fully offline — no external API or internet dependency.
- Easily retrainable by just updating `dataset.csv`.
- Clean, attractive UI built with Streamlit.

**Limitations:**
- Accuracy depends heavily on the size and variety of the dataset —
  with only ~130 sample captions, some overlapping categories (like
  Travel and Nature, or Fitness and Sports) can occasionally be
  confused.
- Cannot understand deep context, sarcasm, or very unusual phrasing
  since it uses TF-IDF (word-based) features, not deep language
  understanding.
- Hashtag suggestions are limited to categories seen during training.

### 11. Future Scope
- Add more captions and categories to improve accuracy.
- Use more advanced NLP techniques (word embeddings, transformers) for
  better context understanding.
- Add trending hashtag updates via a live social media API.
- Support multiple languages (e.g., Hindi/Marathi captions).
- Allow users to give feedback on suggested hashtags to improve the
  model over time (active learning).

### 12. Simple Marathi-English Explanation (For Mam)

"Mam, ha project ek **Machine Learning based web application** ahe je
social media post cha caption (mhanje text) vachun tyacha **category**
(jase Travel, Food, Fashion, ityadi) automatically olkhla jato, ani
tyanusar relevant **hashtags suggest** karto.

Yasathi amhi **TF-IDF** vaprla ahe je text la numbers madhe convert
karto, karan computer la words nahi tar numbers samajtat. Nantar
**Random Forest** naavacha ek simple ani reliable ML algorithm vaprun
model train kela — je multiple chhote decision trees banvun tyanche
"voting" gheun final answer deto, mhanun to ek Decision Tree peksha
jasta accurate asto.

Model la amhi caption cha text ani tyacha category asलेला ek dataset
(csv file) dila, ani tyavarun model ne shikla ki kontya prakarchya
words kontya category madhe yetat. Jevha user navin caption takto, tevha
model tyala category predict karto, ani tya category shी related
hashtags dataset madhun kadhun suggest karto.

He purn project offline chalte, kontahi internet API vaparla nahi, ani
Streamlit vaprun ek sundar, modern web interface banवला ahe jyat user
directly browser madhe caption takun result baghu shakto."

---

## 8. Notes for Demonstration

- Run `train_model.py` once before the demo (or re-run anytime you
  update `dataset.csv`).
- Try the example caption above first to guarantee a clean, correct
  demo result, then try a few original captions live to show the model
  genuinely predicting (not hardcoded).
- Point out that `model.pkl` and `vectorizer.pkl` prove the model was
  actually trained and saved — nothing is hardcoded in `app.py`.
