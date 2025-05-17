import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
df = pd.read_csv('train_timeseries.csv')

# Print initial information
print("Original dataset shape:", df.shape)
print("Original score distribution:\n", df['score'].value_counts(dropna=True).sort_index())

# Step 1: Remove rows with NaN in 'score'
df = df.dropna(subset=['score'])
print("\nAfter removing NaN values:")
print("Dataset shape:", df.shape)

# Step 2: Convert date to numeric features
if 'date' in df.columns:
    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])
    
    # Extract numeric features from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
#    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
#    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Drop the original date column
    df = df.drop('date', axis=1)
    
    print("\nConverted date to numeric features: year, month, day, day_of_week, day_of_year")

# Step 3: Create binary classification dataset
print("\nCreating binary classification dataset...")
print("Before filtering - Dataset shape:", df.shape)

# Keep only instances with score < 1.0 (label as 0) or score > 3.0 (label as 1)
binary_df = df[(df['score'] < 1.0) | (df['score'] > 3.0)].copy()

# Create binary labels: 0 for scores < 1.0, 1 for scores > 3.0
binary_df['score'] = (binary_df['score'] > 3.0).astype(int)

print("\nAfter filtering - Dataset shape:", binary_df.shape)
print("Binary score distribution before balancing:\n", binary_df['score'].value_counts(dropna=True).sort_index())

# Step 4: Balance the dataset
# Get counts of each class
class_counts = binary_df['score'].value_counts()
min_class_count = min(class_counts)

# Create a balanced dataset
balanced_df = pd.DataFrame()

# For each class (0 and 1)
for class_value in [0, 1]:
    # Get all instances of this class
    class_df = binary_df[binary_df['score'] == class_value]
    
    # If this is the majority class, sample down to match the minority class
    if len(class_df) > min_class_count:
        class_df = class_df.sample(n=min_class_count, random_state=42)
    
    # Add to the balanced dataframe
    balanced_df = pd.concat([balanced_df, class_df])

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing - Dataset shape:", balanced_df.shape)
print("Binary score distribution after balancing:\n", balanced_df['score'].value_counts(dropna=True).sort_index())

# Ensure all columns are numeric
for col in balanced_df.columns:
    if balanced_df[col].dtype == 'object':
        try:
            balanced_df[col] = pd.to_numeric(balanced_df[col])
            print(f"Converted '{col}' to numeric.")
        except:
            print(f"WARNING: Could not convert '{col}' to numeric. This column may cause issues.")

# The balanced_df now contains the balanced dataset with binary classification
print(f"\nFinal balanced binary dataset shape: {balanced_df.shape}")
balanced_df.to_csv('train.csv', index=False)
print("\nBalanced binary dataset saved as 'train.csv'")
