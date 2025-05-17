import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('train.csv')
print("Dataset shape:", df.shape)

# Identify the target and features
target = 'score'  # Target column is 'score'

# Check target distribution
print(f"\nTarget '{target}' distribution:")
print(df[target].value_counts().sort_index())

# Identify features (all columns except target)
feature_columns = [col for col in df.columns if col != target]

# Convert any string columns to numeric if needed
for col in feature_columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            print(f"Column {col} could not be converted to numeric, dropping it.")
            feature_columns.remove(col)

# Prepare features and target
X = df[feature_columns].values
y = df[target].values

print(f"\nUsing {len(feature_columns)} features for training.")

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Prepare stratified k-fold cross validation (10 folds = 90% train, 10% test per fold)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store results
fold_accuracies = []
all_y_true = []
all_y_pred = []

# Perform cross-validation
print("\nStarting 10-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train the model on 90% of data
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions on 10% test data
    y_pred = rf_classifier.predict(X_test)
    
    # Store true labels and predictions for confusion matrix
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold+1}/10 - Accuracy: {accuracy:.4f}")

# Calculate and display overall metrics
print("\nCross-validation complete!")
print(f"Average accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")

# Generate confusion matrix
conf_matrix = confusion_matrix(all_y_true, all_y_pred)

# Display confusion matrix as text
print("\nConfusion Matrix:")
print(conf_matrix)

# Display class labels
class_labels = sorted(np.unique(y))
print(f"Class labels: {class_labels}")

# Display classification report
print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred))

# Feature importance from the last fold's model
feature_importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features (from last fold's model):")
print(feature_importance_df.head(10))
