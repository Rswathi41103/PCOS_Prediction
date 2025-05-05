import numpy as np

# Load image features and labels
image_features = np.load('image_features.npy')
image_labels = np.load('image_labels.npy')

# Ensure image features and labels shape are as expected
print("Image features shape:", image_features.shape)
print("Image labels shape:", image_labels.shape)

# Load data features and labels
data_features = np.load('data_features.npy')
data_labels = np.load('data_labels.npy')

# Ensure data features and labels shape are as expected
print("Data features shape:", data_features.shape)
print("Data labels shape:", data_labels.shape)

# Check if number of samples are the same in image and data labels
if len(image_labels) != len(data_labels):
    raise ValueError("The number of samples in image labels and data labels do not match!")

# Combine image features and data features
if image_features.ndim == 1:
    image_features = np.expand_dims(image_features, axis=0)
if data_features.ndim == 1:
    data_features = np.expand_dims(data_features, axis=0)

combined_features = np.concatenate((image_features, data_features), axis=1)
combined_labels = image_labels  # or use data_labels if they are the same

# Check the shape of the combined features and labels
print("Combined features shape:", combined_features.shape)
print("Combined labels shape:", combined_labels.shape)

# Save the combined features and labels
np.save('combined_features.npy', combined_features)
np.save('combined_labels.npy', combined_labels)

print("Combined features and labels saved to disk.")
