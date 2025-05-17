import pandas as pd
import numpy as np
from sklearn.utils import resample
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
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Drop the original date column
    df = df.drop('date', axis=1)
    
    print("\nConverted date to numeric features: year, month, day, day_of_week, day_of_year")

# Step 3: Convert score to 6 integer classes (0 to 5)
# First, ensure score is within 0-5 range
df = df[df['score'] >= 0]
df = df[df['score'] <= 5]

# Convert to integer classes (0, 1, 2, 3, 4, 5)
df['score'] = df['score'].apply(lambda x: int(round(x)))  # Round to nearest integer
df['score'] = df['score'].astype(int)  # Ensure it's stored as integer type

print("\nVerifying score is integer type:", df['score'].dtype)
print("Unique values in score:", df['score'].unique())

# Step 4: Check the distribution of classes
class_counts = df['score'].value_counts().sort_index()
print("\nClass distribution after conversion:")
print(class_counts)

# Step 5: Balance the classes by selecting the same number of instances from each class
min_class_count = class_counts.min()
print(f"\nSmallest class has {min_class_count} instances")

# Create a balanced dataset
balanced_df = pd.DataFrame()

for class_value in range(6):  # 0 to 5
    class_df = df[df['score'] == class_value]
    
    # If the class exists in our dataset
    if not class_df.empty:
        # Randomly sample the minimum number of instances
        sampled_df = resample(class_df,
                             replace=False,  # sample without replacement
                             n_samples=min_class_count,
                             random_state=42)  # reproducible results
        balanced_df = pd.concat([balanced_df, sampled_df])

# Check the balanced distribution
balanced_class_counts = balanced_df['score'].value_counts().sort_index()
print("\nBalanced class distribution:")
print(balanced_class_counts)

# Ensure all columns are numeric
for col in balanced_df.columns:
    if balanced_df[col].dtype == 'object':
        try:
            balanced_df[col] = pd.to_numeric(balanced_df[col])
            print(f"Converted '{col}' to numeric.")
        except:
            print(f"WARNING: Could not convert '{col}' to numeric. This column may cause issues.")

# The balanced_df now contains your balanced dataset
print(f"\nFinal balanced dataset shape: {balanced_df.shape}")
balanced_df.to_csv('train.csv', index=False)
print("\nBalanced dataset saved as 'train.csv'")
