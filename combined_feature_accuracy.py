import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load combined features and labels
combined_features = np.load('combined_features.npy')
combined_labels = np.load('combined_labels.npy')

# Ensure combined features and labels shape are as expected
print("Combined features shape:", combined_features.shape)
print("Combined labels shape:", combined_labels.shape)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_labels, test_size=0.2, random_state=42)

# Train a Gradient Boosting classifier
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
classifier.fit(X_train, y_train)

# Predict on the training set
y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict on the test set
y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
