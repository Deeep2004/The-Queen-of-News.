import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


train_data_path = "train.csv"  
df = pd.read_csv(train_data_path, names=["index", "category", "headline"], skiprows=1)

# Preprocessing
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


# Use a subset of training data for GridSearchCV
subset_size = 10000  # Define the number of samples to use
subset_texts, _, subset_labels, _ = train_test_split(
    train_texts, train_labels, train_size=subset_size, stratify=train_labels, random_state=42
)

# Transform the subset of data
X_subset = vectorizer.transform(subset_texts)

# Define parameter grid for Logistic Regression
param_grid = {
    'C': [10.0 ** i for i in range(-5, 6)],
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced']
}

# Set up the model and GridSearchCV
model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)  # Reduced CV folds to 3
grid_search.fit(X_subset, subset_labels)

# Best parameters and score
best_C = grid_search.best_params_['C']
best_penalty = grid_search.best_params_['penalty']
best_score = grid_search.best_score_

print(f"Best Parameters: C={best_C}, Penalty={best_penalty}")
print(f"Best Cross-Validation Accuracy: {best_score}")

# Train the optimized model on the full training data
optimized_model = LogisticRegression(C=best_C, penalty=best_penalty, class_weight='balanced',
                                      solver='liblinear', max_iter=1000, random_state=42)
optimized_model.fit(X_train, train_labels)

# Evaluate the optimized model
val_predictions = optimized_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
print("\nClassification Report:\n", classification_report(val_labels, val_predictions, target_names=label_encoder.classes_))

# # Save the optimized model, vectorizer, and label encoder
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
# joblib.dump(optimized_model, "optimized_logistic_regression_model.pkl")
# joblib.dump(label_encoder, "label_encoder.pkl")

print("Optimized model, vectorizer, and label encoder saved.")

# File paths
input_file = "5%_abcnews-date-text.csv"  #
output_file = "prediction_LRV1.2.csv"  # File to save predictions

# Reload saved model, vectorizer, and label encoder
vectorizer = joblib.load("tfidf_vectorizer.pkl")
optimized_model = joblib.load("optimized_logistic_regression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define a batch processing function
def predict_batch(batch):
    # Extract the headline column (ignoring the date column)
    headlines = batch.iloc[:, 1]  # Assumes the second column contains headlines
    # Transform headlines using the TF-IDF vectorizer
    X_batch = vectorizer.transform(headlines)
    # Predict categories
    predictions = optimized_model.predict(X_batch)
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
