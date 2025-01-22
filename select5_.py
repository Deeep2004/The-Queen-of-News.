import pandas as pd

# Load the CSV file
file_path = 'abcnews-date-text.csv' 
data = pd.read_csv(file_path)

# Calculate 5% of the total number of rows
sample_size = int(len(data) * 0.05)

# Randomly select 5% of the rows
random_sample = data.sample(n=sample_size, random_state=1)  # random_state for reproducibility

# Save the random sample to a new CSV file
output_file_path = '5%_abcnews-date-text.csv'  # Specify the output file name
random_sample.to_csv(output_file_path, index=False)

print(f"Random sample saved to {output_file_path}")