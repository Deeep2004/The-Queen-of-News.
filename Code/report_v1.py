import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load the CSV file, skipping the first 15 lines
file_path = "prediction_LRV2.0.csv"  # Replace with your file path
df = pd.read_csv(file_path, skiprows=15, header=None)

# Assigning column names
df.columns = ["Date", "Headline", "Category", "Subcategory"]

# Convert the date to a datetime format for easier manipulation
df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')

# Extracting the year for analysis
df["Year"] = df["Date"].dt.year

# 1. Total number of headlines per category
category_counts = df["Category"].value_counts()

# 2. Top 5 categories
top_5_categories = category_counts.head(5)

# 3. Density of headlines over time (yearly trend)
yearly_trend = df["Year"].value_counts().sort_index()

# 4. Most popular category for each 5-year period
df["5-Year Period"] = (df["Year"] // 5) * 5
popular_categories_by_period = df.groupby("5-Year Period")["Category"].agg(lambda x: Counter(x).most_common(1)[0])

# Generate the report
print("=== Total Headlines Per Category ===")
print(category_counts)
print("\n=== Top 5 Categories ===")
print(top_5_categories)
print("\n=== Yearly Trend of Headlines ===")
print(yearly_trend)
print("\n=== Most Popular Category for Each 5-Year Period ===")
print(popular_categories_by_period)

# Visualizing results
# 1. Bar chart for total headlines per category
category_counts.plot(kind='bar', figsize=(10, 6), title="Total Headlines Per Category")
plt.xlabel("Category")
plt.ylabel("Number of Headlines")
plt.show()

# 2. Top 5 categories
top_5_categories.plot(kind='bar', figsize=(8, 5), color='orange', title="Top 5 Categories")
plt.xlabel("Category")
plt.ylabel("Number of Headlines")
plt.show()

# 3. Line chart for yearly trend
yearly_trend.plot(kind='line', figsize=(12, 6), marker='o', title="Yearly Trend of Headlines")
plt.xlabel("Year")
plt.ylabel("Number of Headlines")
plt.grid()
plt.show()

# 4. Most popular category for each 5-year period
popular_categories_by_period.plot(kind='bar', figsize=(10, 5), color='green', title="Most Popular Category (Every 5 Years)")
plt.xlabel("5-Year Period")
plt.ylabel("Category")
plt.show()
