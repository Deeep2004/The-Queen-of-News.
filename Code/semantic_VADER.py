import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Load CSV file, skipping first 15 lines
file_path = "prediction_LRV2.0.csv"  # Replace with your file path
df = pd.read_csv(file_path, skiprows=15, header=None)

# Assign column names
df.columns = ["Date", "Headline", "Category", "Subcategory"]

# Convert date to datetime and extract year
df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
df["Year"] = df["Date"].dt.year

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate VADER sentiment scores
def get_sentiment_scores(headline):
    sentiment = sia.polarity_scores(headline)
    return sentiment["compound"]

# Apply the function to each headline
df["Sentiment_Score"] = df["Headline"].apply(get_sentiment_scores)

# Add sentiment labels (Positive, Neutral, Negative)
def label_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment_Label"] = df["Sentiment_Score"].apply(label_sentiment)

# 1. Yearly average sentiment score by category
yearly_sentiment = df.groupby(["Year", "Category"])["Sentiment_Score"].mean().unstack()

# 2. Sentiment distribution over time (positive, neutral, negative counts)
yearly_sentiment_counts = df.groupby(["Year", "Category", "Sentiment_Label"]).size().unstack()

# Save results to a file
output_file = "vader_sentiment_analysis_report.txt"
with open(output_file, "w") as f:
    f.write("=== Average Sentiment Score Per Year (By Category) ===\n")
    f.write(yearly_sentiment.to_string() + "\n\n")
    
    f.write("=== Sentiment Distribution Over Years ===\n")
    f.write(yearly_sentiment_counts.to_string() + "\n\n")

print(f"Sentiment analysis saved to {output_file}")

# 3. Visualization
# Plot average sentiment score per year for top 5 categories
top_5_categories = df["Category"].value_counts().head(5).index
yearly_sentiment[top_5_categories].plot(figsize=(12, 6), title="Yearly Sentiment Trend (Top 5 Categories)")
plt.xlabel("Year")
plt.ylabel("Average Sentiment Score")
plt.grid()
plt.show()

# Sentiment distribution for a single category over time (e.g., Politics)
example_category = "POLITICS"
example_sentiment = yearly_sentiment_counts.loc[:, example_category]
example_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6), title=f"Sentiment Distribution Over Years: {example_category}")
plt.xlabel("Year")
plt.ylabel("Number of Headlines")
plt.legend(title="Sentiment")
plt.grid()
plt.show()
