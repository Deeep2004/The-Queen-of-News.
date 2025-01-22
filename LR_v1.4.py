from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
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

# Step 1: Use a subset of data for finding candidate C values
subset_size = 5000  # Define a smaller subset for faster processing
subset_texts, _, subset_labels, _ = train_test_split(
    train_texts, train_labels, train_size=subset_size, stratify=train_labels, random_state=42
)

# Transform the subset of data
X_subset = vectorizer.transform(subset_texts)
X_val_subset = X_val  # Still validate on the full validation set

# Find candidate C values with accuracy > 0.4
candidate_C_values = []
for C in [10.0 ** i for i in range(-3, 3)]:
    # Train a logistic regression model on the subset
    model = LogisticRegression(C=C, penalty='l2', class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_subset, subset_labels)
    
    # Evaluate on the validation set
    val_predictions = model.predict(X_val)
    acc = accuracy_score(val_labels, val_predictions)
    
    # Add C to candidate list if accuracy > 0.4
    if acc > 0.4:
        candidate_C_values.append(C)

print("Candidate C values with accuracy > 0.4:", candidate_C_values)

# Step 2: Predict on the unseen data using each candidate C value
all_predictions = defaultdict(list)  # Store predictions for each C value by headline index

input_file = "5%_abcnews-date-text.csv"  
output_file = "prediction_LRV1.4.csv"  # File to save predictions
chunk_size = 10000  # Number of rows to process at a time

for C in candidate_C_values:
    # Train model on the full training dataset
    model = LogisticRegression(C=C, penalty='l2', class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_train, train_labels)
    
    # Predict on the unseen data in batches
    with pd.read_csv(input_file, chunksize=chunk_size, header=None) as reader:
        for chunk_idx, chunk in enumerate(reader):
            # Extract headlines
            headlines = chunk.iloc[:, 1]  # Assumes the second column contains headlines
            X_batch = vectorizer.transform(headlines)
            # Predict using the model
            predictions = model.predict(X_batch)
            
            # Store predictions per headline
            for idx, pred in enumerate(predictions):
                all_predictions[chunk_idx * chunk_size + idx].append(pred)

# # Step 3: Evaluate the accuracy of multi-category predictions
# correct_predictions = 0
# total_samples = len(val_labels)  # Assumes unseen data has validation labels available

# for idx, true_label in enumerate(val_labels):  # Compare to true labels
#     predicted_labels = all_predictions[idx]
#     if true_label in predicted_labels:
#         correct_predictions += 1

# accuracy = correct_predictions / total_samples
# print("Accuracy considering multiple predictions per headline:", accuracy)

# Step 4: Save predictions for the unseen data
output_file = "prediction_multi_label.csv"
with pd.read_csv(input_file, chunksize=chunk_size, header=None) as reader:
    with open(output_file, 'w') as f:
        for chunk_idx, chunk in enumerate(reader):
            chunk["predicted_categories"] = [
                ", ".join(label_encoder.inverse_transform(all_predictions[chunk_idx * chunk_size + idx]))
                for idx in range(len(chunk))
            ]
            chunk.to_csv(f, index=False, header=f.tell() == 0)

print(f"Predictions saved to {output_file}")
