import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

train_data_path = "train.csv"  
df = pd.read_csv(train_data_path, names=["index", "category", "headline"], skiprows=1)
df["category"] = df["category"].str.strip() 
df["headline"] = df["headline"].str.strip()
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["category"])
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["headline"], df["category_encoded"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_labels)
val_probs = model.predict_proba(X_val)
val_top2 = [label_encoder.inverse_transform(probs.argsort()[-2:][::-1]) for probs in val_probs]

val_true_labels = label_encoder.inverse_transform(val_labels)
top2_accuracy = sum([1 if val_true_labels[i] in val_top2[i] else 0 for i in range(len(val_labels))]) / len(val_labels)
print("Validation Top-2 Accuracy:", top2_accuracy)
print("Model, vectorizer, and label encoder saved.")

input_file = "5%_abcnews-date-text.csv" 
output_file = "prediction_LRV2.0.csv"  
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_top2_batch(batch):
    headlines = batch.iloc[:, 1]  
    X_batch = vectorizer.transform(headlines)

    probs = model.predict_proba(X_batch)
    top2_categories = [
        label_encoder.inverse_transform(probs[i].argsort()[-2:][::-1]) for i in range(len(probs))
    ]
    return top2_categories

chunk_size = 10000  
with pd.read_csv(input_file, chunksize=chunk_size, header=None) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")
        top2_predictions = predict_top2_batch(chunk)  
        chunk["top1_category"] = [pred[0] for pred in top2_predictions]
        chunk["top2_category"] = [pred[1] for pred in top2_predictions]
        chunk.to_csv(output_file, mode="a", index=False, header=False)

print("Predictions with top 2 categories completed! All results saved.")
