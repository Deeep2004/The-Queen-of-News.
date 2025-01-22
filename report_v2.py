import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load the CSV file, skipping the first 15 lines
file_path = "prediction_LRV2.0.csv"  # Replace with your file path
df = pd.read_csv(file_path, skiprows=15, header=None)

# Assigning column names
df.columns = ["Date", "Headline", "Category", "Subcategory"]

# Convert the date to a datetime format
df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')

# Extracting additional time information
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 1. Total number of headlines per category
category_counts = df["Category"].value_counts()

# 2. Top 5 categories
top_5_categories = category_counts.head(5)

# 3. Density of headlines over time
yearly_trend = df["Year"].value_counts().sort_index()

# 4. Most popular category for each 5-year period
df["5-Year Period"] = (df["Year"] // 5) * 5
popular_categories_by_period = df.groupby("5-Year Period")["Category"].agg(lambda x: Counter(x).most_common(1)[0])

# 5. Headline length analysis
df["Headline_Length"] = df["Headline"].apply(len)
headline_length_stats = df.groupby("Category")["Headline_Length"].agg(['mean', 'min', 'max'])

# 6. Subcategory trends
subcategory_counts = df["Subcategory"].value_counts()

# 7. Seasonal trends
monthly_trend = df["Month"].value_counts().sort_index()

# 8. Most frequent words in headlines by category
stopwords = set(["the", "and", "a", "to", "of", "in", "for", "on", "with", "at", "by", "an", "this", "is", "that"])
def clean_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stopwords]

frequent_words_by_category = {}
for category in df["Category"].unique():
    category_headlines = df[df["Category"] == category]["Headline"]
    words = []
    for headline in category_headlines:
        words.extend(clean_text(headline))
    frequent_words_by_category[category] = Counter(words).most_common(10)

# Save results to a text file
output_file = "analysis_report.txt"
with open(output_file, "w") as f:
    f.write("=== Total Headlines Per Category ===\n")
    f.write(category_counts.to_string() + "\n\n")
    
    f.write("=== Top 5 Categories ===\n")
    f.write(top_5_categories.to_string() + "\n\n")
    
    f.write("=== Yearly Trend of Headlines ===\n")
    f.write(yearly_trend.to_string() + "\n\n")
    
    f.write("=== Most Popular Category for Each 5-Year Period ===\n")
    for period, (category, count) in popular_categories_by_period.items():
        f.write(f"{period}: {category} ({count} headlines)\n")
    f.write("\n")
    
    f.write("=== Headline Length Analysis (Per Category) ===\n")
    f.write(headline_length_stats.to_string() + "\n\n")
    
    f.write("=== Subcategory Trends ===\n")
    f.write(subcategory_counts.to_string() + "\n\n")
    
    f.write("=== Monthly Trends ===\n")
    f.write(monthly_trend.to_string() + "\n\n")
    
    f.write("=== Most Frequent Words by Category ===\n")
    for category, words in frequent_words_by_category.items():
        f.write(f"\n{category}:\n")
        for word, count in words:
            f.write(f"  {word}: {count}\n")

print(f"Analysis saved to {output_file}")
