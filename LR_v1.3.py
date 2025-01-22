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
subset_size = 5000  # Define a smaller subset size for faster processing
subset_texts, _, subset_labels, _ = train_test_split(
    train_texts, train_labels, train_size=subset_size, stratify=train_labels, random_state=42
)

# Transform the subset of data
X_subset = vectorizer.transform(subset_texts)
X_val_subset = X_val  # Still validate on the full validation set

# Find candidate C values
candidate_C_values = []
accuracies = {}

for C in [10.0 ** i for i in range(-5, 6)]:  # Range of C values
    # Train logistic regression model
    model = LogisticRegression(C=C, penalty='l2', class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_subset, subset_labels)
    
    # Evaluate on the full validation set
    val_predictions = model.predict(X_val_subset)
    acc = accuracy_score(val_labels, val_predictions)
    accuracies[C] = acc
    
    # Add C to candidate list if accuracy > 0.4
    if acc > 0.4:
        candidate_C_values.append(C)

print("Candidate C values with accuracy > 0.4:", candidate_C_values)

# Step 2: Train models on full dataset using candidate C values and predict
all_predictions = []  # Store predictions from all models
input_file = "5%_abcnews-date-text.csv"  
output_file = "prediction_LRV1.3.csv"  # File to save predictions
chunk_size = 10000  # Number of rows to process at a time

for C in candidate_C_values:
    # Train a new model using the full training set
    model = LogisticRegression(C=C, penalty='l2', class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_train, train_labels)
    
    # Predict on the unseen data in batches
    unseen_predictions = []
    with pd.read_csv(input_file, chunksize=chunk_size, header=None) as reader:
        for chunk in reader:
            # Extract headlines
            headlines = chunk.iloc[:, 1]  # Assumes the second column contains headlines
            X_batch = vectorizer.transform(headlines)
            # Predict using the model
            predictions = model.predict(X_batch)
            unseen_predictions.extend(predictions)
    
    all_predictions.append(unseen_predictions)

# Step 3: Aggregate predictions using majority voting
all_predictions = np.array(all_predictions)  # Shape: (len(candidate_C_values), num_samples)

final_predictions = []
for i in range(all_predictions.shape[1]):  # Iterate over all headlines
    headline_predictions = all_predictions[:, i]
    most_common = Counter(headline_predictions).most_common(1)[0][0]
    final_predictions.append(most_common)

# Decode final predictions
final_categories = label_encoder.inverse_transform(final_predictions)

# Step 4: Save final predictions
df_input = pd.read_csv(input_file, header=None)
df_input["final_predicted_category"] = final_categories
df_input.to_csv(output_file, index=False)
print("Final predictions saved to", output_file)


from sklearn.metrics import accuracy_score
# Step 1: Predict training set using each candidate C value
train_predictions = []  # Store predictions for training data from all models

for C in candidate_C_values:
    # Train a model with the specific C value
    model = LogisticRegression(C=C, penalty='l2', class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_subset, subset_labels)
    # Predict on the training set
    predictions = model.predict(X_val_subset)
    train_predictions.append(predictions)

# Step 2: Aggregate predictions for majority voting
train_predictions = np.array(train_predictions)  # Shape: (len(candidate_C_values), num_train_samples)

final_train_predictions = []
for i in range(train_predictions.shape[1]):  # Iterate over all training samples
    sample_predictions = train_predictions[:, i]
    most_common = Counter(sample_predictions).most_common(1)[0][0]
    final_train_predictions.append(most_common)

# Step 3: Calculate accuracy of majority-voted predictions
final_train_accuracy = accuracy_score(val_labels, final_train_predictions)

print("Accuracy of the combined prediction on the training set:", final_train_accuracy)
