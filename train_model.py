import pandas as pd
import random
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ================= CLEAN FUNCTION =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ================= LOAD DATA =================
df_pdf = pd.read_csv("dataset.csv")

df_kaggle = pd.read_csv("AI_Resume_Screening.csv")
df_kaggle = df_kaggle.rename(columns={
    "Skills": "text",
    "Job Role": "role"
})
df_kaggle = df_kaggle[["text", "role"]]

# ================= MERGE =================
df = pd.concat([df_pdf, df_kaggle], ignore_index=True)

# ================= CLEAN =================
df = df.dropna()
df = df[df["text"].str.len() > 20]

df["text"] = df["text"].apply(clean_text)

# ================= ROLE NORMALIZATION =================
df["role"] = df["role"].replace({
    "Data_Scientist": "Data Scientist",
    "ML_Engineer": "AI Researcher",
    "Java_Developer": "Software Engineer",
    "Web_Developer": "Software Engineer",
    "DevOps_Engineer": "Software Engineer",
    "Database_Admin": "Software Engineer"
})

print("\nAfter Role Fix:\n")
print(df["role"].value_counts())

# ================= SYNTHETIC DATA =================
templates = [
    "experience in {} and {} using {}",
    "worked on {} projects with {} and {}",
    "strong knowledge of {} and {} in {}",
]

skills_map = {
    "Data Scientist": ["python", "pandas", "machine learning", "numpy"],
    "AI Researcher": ["deep learning", "nlp", "tensorflow", "pytorch"],
    "Cybersecurity Analyst": ["network security", "penetration testing", "cryptography"],
    "Software Engineer": ["java", "api", "system design", "web development"]
}

synthetic_rows = []

for role, skills in skills_map.items():
    for _ in range(25):  # small augmentation
        text = random.choice(templates).format(
            random.choice(skills),
            random.choice(skills),
            random.choice(skills)
        )
        synthetic_rows.append({"text": text, "role": role})

df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

print("\nFinal Data Distribution:\n")
print(df["role"].value_counts())

# ================= FEATURES =================
X = df["text"]
y = df["role"]

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)

X_vectors = vectorizer.fit_transform(X)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= MODELS =================
models = {
    "svm": LinearSVC(),
    "lr": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced")
}

trained_models = {}

best_model = None
best_score = 0
best_name = ""

# ================= TRAIN & EVALUATE =================
for name, model in models.items():
    print(f"\n===== {name.upper()} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    trained_models[name] = model

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

# ================= SAVE =================
pickle.dump(trained_models, open("models.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n🏆 BEST MODEL:", best_name)
print("BEST ACCURACY:", best_score)

print("\n✅ ALL MODELS SAVED SUCCESSFULLY!")