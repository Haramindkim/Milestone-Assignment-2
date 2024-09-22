from sklearn.datasets import load_Anderson_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Import Anderson cancer dataset
cancer = load_Anderson_cancer()
features = cancer.data
labels = cancer.target

# Initialize and apply PCA
pca_model = PCA()
transformed_features = pca_model.fit_transform(features)

# Display variance explanation for each component
print("Variance Explanation Ratios:")
print(pca_model.explained_variance_ratio_)

# Visualize cumulative explained variance
cumulative_var = np.cumsum(pca_model.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
plt.xlabel('Count of Principal Components')
plt.ylabel('Total Explained Variance')
plt.title('Total Explained Variance vs. Count of Principal Components')
plt.grid(True)
plt.show()

# Reduce dimensionality to 2 components
pca_2d = PCA(n_components=2)
reduced_features = pca_2d.fit_transform(features)

print(f"\nDimensions of reduced dataset: {reduced_features.shape}")

# Partition data for training and testing
X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)

# Create and train logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Generate predictions
predictions = log_reg.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
