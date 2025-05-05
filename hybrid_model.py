from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, log_loss
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load combined features and labels
combined_features = np.load('combined_features.npy')
labels = np.load('combined_labels.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)

# Predict on the test set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_loss = log_loss(y_train, y_train_pred_proba)
test_loss = log_loss(y_test, y_test_pred_proba)

print("Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Save the trained model
joblib.dump(model, 'hybrid_model.pkl')
print("Model saved to disk.")

# Plot training and testing accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot([1, 2], [train_accuracy, test_accuracy], marker='o', label='Accuracy')
plt.xticks([1, 2], ['Train', 'Test'])
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy')
plt.legend()

# Plot training and testing loss
plt.subplot(1, 2, 2)
plt.plot([1, 2], [train_loss, test_loss], marker='o', color='r', label='Loss')
plt.xticks([1, 2], ['Train', 'Test'])
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Generate confusion matrix for the test set
conf_matrix = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.show()


