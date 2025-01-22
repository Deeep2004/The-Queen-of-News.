import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

train_data_path = "train.csv" 
df = pd.read_csv(train_data_path, names=["index", "category", "headline"], skiprows=1)
df["category"] = df["category"].str.strip()  # Remove extra spaces
df["headline"] = df["headline"].str.strip()

# Encode labels
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["category"])

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["headline"], df["category_encoded"], test_size=0.2, random_state=42
)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_labels)

val_probs = model.predict_proba(X_val)

# Get the top 3 predicted categories for each headline
val_top3 = [probs.argsort()[-3:][::-1] for probs in val_probs]  # Indices of top 3 predictions
val_top3_labels = [label_encoder.inverse_transform(indices) for indices in val_top3]  # Convert to category names

# Ground truth labels
val_true_labels = label_encoder.inverse_transform(val_labels)

# Compute top-3 accuracy
top3_correct_count = sum(
    1 if val_true_labels[i] in val_top3_labels[i] else 0 for i in range(len(val_labels))
)
top3_accuracy = top3_correct_count / len(val_labels)

print(f"Validation Top-3 Accuracy: {top3_accuracy:.2f}")

# Create a detailed classification report for top-1 predictions
val_top1_predictions = model.predict(X_val)
# print("\nTop-1 Classification Report:\n")
# print(classification_report(val_labels, val_top1_predictions, target_names=label_encoder.classes_))

# Save the trained model, vectorizer, and label encoder
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model, vectorizer, and label encoder saved.")


input_file = "5%_abcnews-date-text.csv"  
output_file = "prediction_LRV2.1.csv"  # File to save predictions

# Reload saved model, vectorizer, and label encoder
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define a batch processing function
def predict_top3_batch(batch):
    # Extract the headline column (ignoring the date column)
    headlines = batch.iloc[:, 1]  # Assumes the second column contains headlines
    # Transform headlines using the TF-IDF vectorizer
    X_batch = vectorizer.transform(headlines)
    # Get top 3 predictions
    probs = model.predict_proba(X_batch)
    top3_categories = [
        label_encoder.inverse_transform(probs[i].argsort()[-3:][::-1]) for i in range(len(probs))
    ]
    return top3_categories

# Process the file in chunks
chunk_size = 10000  # Number of rows to process at a time
with pd.read_csv(input_file, chunksize=chunk_size, header=None) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")
        top3_predictions = predict_top3_batch(chunk)  # Get top 3 predictions for this chunk
        # Add predictions to the chunk
        chunk["top1_category"] = [pred[0] for pred in top3_predictions]
        chunk["top2_category"] = [pred[1] for pred in top3_predictions]
        chunk["top3_category"] = [pred[2] for pred in top3_predictions]
        # Save intermediate results
        chunk.to_csv(output_file, mode="a", index=False, header=False)

print("Predictions with top 3 categories completed! All results saved.")
