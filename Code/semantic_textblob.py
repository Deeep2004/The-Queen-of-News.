import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load CSV file, skipping first 15 lines
file_path = "prediction_LRV2.0.csv"  # Replace with your file path
df = pd.read_csv(file_path, skiprows=15, header=None)

# Assign column names
df.columns = ["Date", "Headline", "Category", "Subcategory"]

# Convert date to datetime and extract year
df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
df["Year"] = df["Date"].dt.year

# Function to calculate sentiment scores using TextBlob
def get_sentiment_score(headline):
    blob = TextBlob(headline)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Apply the function to each headline
df["Polarity"], df["Subjectivity"] = zip(*df["Headline"].apply(get_sentiment_score))

# 1. Yearly average polarity and subjectivity by category
yearly_polarity = df.groupby(["Year", "Category"])["Polarity"].mean().unstack()
yearly_subjectivity = df.groupby(["Year", "Category"])["Subjectivity"].mean().unstack()

# Save results to a file
output_file = "textblob_sentiment_analysis_report.txt"
with open(output_file, "w") as f:
    f.write("=== Average Polarity Per Year (By Category) ===\n")
    f.write(yearly_polarity.to_string() + "\n\n")
    
    f.write("=== Average Subjectivity Per Year (By Category) ===\n")
    f.write(yearly_subjectivity.to_string() + "\n\n")

print(f"Sentiment analysis saved to {output_file}")

# Visualization
# Plot average polarity per year for top 5 categories
top_5_categories = df["Category"].value_counts().head(5).index
yearly_polarity[top_5_categories].plot(figsize=(12, 6), title="Yearly Polarity Trend (Top 5 Categories)")
plt.xlabel("Year")
plt.ylabel("Average Polarity")
plt.grid()
plt.show()

# Plot average subjectivity per year for top 5 categories
yearly_subjectivity[top_5_categories].plot(figsize=(12, 6), title="Yearly Subjectivity Trend (Top 5 Categories)", color='orange')
plt.xlabel("Year")
plt.ylabel("Average Subjectivity")
plt.grid()
plt.show()
