import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('train.csv')

# Verify the target variable
print("\nTarget variable 'score' distribution:")
print(df['score'].value_counts().sort_index())

# Convert score to int if needed
if df['score'].dtype != 'int64':
    df['score'] = df['score'].astype(int)
    print("Converted 'score' to integer type")

# Separate features and target
X = df.drop('score', axis=1)
y = df['score']

print(f"\nDataset shape: {X.shape}")

# Define the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Define cross-validation strategy - changed to 5-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- Method 1: PCA Feature Selection ----------
print("\n\n================= PCA Feature Selection =================")

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=5)  # Using 5 components
X_pca = pca.fit_transform(X_scaled)

# Find the top 5 features based on their contribution to the principal components
feature_importance = np.sum(np.abs(pca.components_), axis=0)
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Sort features by importance and select top 5
top_pca_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
top_pca_feature_names = [feature[0] for feature in top_pca_features]

print("Top 5 features by PCA contribution:")
for i, (feature, importance) in enumerate(top_pca_features, 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Create reduced dataset with top 5 PCA features
X_pca_top_features = X[top_pca_feature_names]

# Conduct 5-fold cross-validation with Random Forest using PCA top features
print("\nConducting 5-fold cross-validation with Random Forest using PCA top 5 features...")
pca_scores = cross_val_score(clf, X_pca_top_features, y, cv=cv, scoring='accuracy')
print(f"PCA - 5-fold CV Accuracy: {pca_scores.mean():.4f} ± {pca_scores.std():.4f}")

# ---------- Method 2: Spearman Correlation Feature Selection ----------
print("\n\n================= Spearman Correlation Feature Selection =================")

# Calculate Spearman correlation using scipy library
spearman_correlations = []
for col in X.columns:
    correlation, _ = spearmanr(X[col], y)
    spearman_correlations.append((col, abs(correlation)))

# Sort and get top 5 features
spearman_sorted = sorted(spearman_correlations, key=lambda x: x[1], reverse=True)
top_features_spearman = [item[0] for item in spearman_sorted[:5]]

print("Top 5 features by Spearman correlation:")
for i, (feature, correlation) in enumerate(spearman_sorted[:5], 1):
    print(f"{i}. {feature}: {correlation:.4f}")

# Create reduced dataset with selected features
X_spearman = X[top_features_spearman]

# Conduct 5-fold cross-validation with Random Forest using Spearman reduced dataset
print("\nConducting 5-fold cross-validation with Random Forest using Spearman reduced dataset...")
spearman_scores = cross_val_score(clf, X_spearman, y, cv=cv, scoring='accuracy')
print(f"Spearman - 5-fold CV Accuracy: {spearman_scores.mean():.4f} ± {spearman_scores.std():.4f}")

# ---------- Method 3: Kendall Correlation Feature Selection ----------
print("\n\n================= Kendall Correlation Feature Selection =================")

# Calculate Kendall correlation using scipy library
kendall_correlations = []
for col in X.columns:
    correlation, _ = kendalltau(X[col], y)
    kendall_correlations.append((col, abs(correlation)))

# Sort and get top 5 features
kendall_sorted = sorted(kendall_correlations, key=lambda x: x[1], reverse=True)
top_features_kendall = [item[0] for item in kendall_sorted[:5]]

print("Top 5 features by Kendall correlation:")
for i, (feature, correlation) in enumerate(kendall_sorted[:5], 1):
    print(f"{i}. {feature}: {correlation:.4f}")

# Create reduced dataset with selected features
X_kendall = X[top_features_kendall]

# Conduct 5-fold cross-validation with Random Forest using Kendall reduced dataset
print("\nConducting 5-fold cross-validation with Random Forest using Kendall reduced dataset...")
kendall_scores = cross_val_score(clf, X_kendall, y, cv=cv, scoring='accuracy')
print(f"Kendall - 5-fold CV Accuracy: {kendall_scores.mean():.4f} ± {kendall_scores.std():.4f}")

# ---------- Summary ----------
print("\n\n================= Summary of Results =================")
print(f"PCA (5 features):      {pca_scores.mean():.4f} ± {pca_scores.std():.4f}")
print(f"Spearman (5 features): {spearman_scores.mean():.4f} ± {spearman_scores.std():.4f}")
print(f"Kendall (5 features):  {kendall_scores.mean():.4f} ± {kendall_scores.std():.4f}")

# Save the reduced datasets
X_pca_top_features['score'] = y
X_pca_top_features.to_csv('pca_top5_features.csv', index=False)
print("\nSaved PCA top 5 features dataset to 'pca_top5_features.csv'")

X_spearman['score'] = y
X_spearman.to_csv('spearman_top5_features.csv', index=False)
print("Saved Spearman top 5 features dataset to 'spearman_top5_features.csv'")

X_kendall['score'] = y
X_kendall.to_csv('kendall_top5_features.csv', index=False)
print("Saved Kendall top 5 features dataset to 'kendall_top5_features.csv'")