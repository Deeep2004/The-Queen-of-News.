import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
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

val_predictions = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
print("\nClassification Report:\n", classification_report(val_labels, val_predictions, target_names=label_encoder.classes_))

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model, vectorizer, and label encoder saved.")

input_file = "5%_abcnews-date-text.csv"  
output_file = "prediction_LRV1.csv"  # File to save predictions

# Reload saved model, vectorizer, and label encoder
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define a batch processing function
def predict_batch(batch):
    headlines = batch.iloc[:, 1]  # Assumes the second column contains headlines
    # Transform headlines using the TF-IDF vectorizer
    X_batch = vectorizer.transform(headlines)
    # Predict categories
    predictions = model.predict(X_batch)
    # Decode category labels
    return label_encoder.inverse_transform(predictions)

# Process the file in chunks
chunk_size = 10000  # Number of rows to process at a time
with pd.read_csv(input_file, chunksize=chunk_size, header=None) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")
        chunk_predictions = predict_batch(chunk)  # Predict for this chunk
        # Append predictions to the results with the original data
        chunk["predicted_category"] = chunk_predictions
        # Save intermediate results
        chunk.to_csv(output_file, mode="a", index=False, header=False)

print("Predictions completed! All results saved.")
