import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load image features and labels
image_labels = np.load('image_labels.npy')

# Load clinical data labels
clinical_df = pd.read_csv('PCOS_data.csv')
clinical_labels = clinical_df['PCOS (Y/N)'].replace({'Y': 1, 'N': 0}).values

# Ensure the number of labels matches
if len(image_labels) != len(clinical_labels):
    raise ValueError("The number of samples in image labels and clinical labels do not match!")

# Combine labels (if needed)
combined_labels = np.concatenate((image_labels, clinical_labels))

# Create a DataFrame for distribution analysis
combined_df = pd.DataFrame(combined_labels, columns=['PCOS (Y/N)'])

# Count the occurrences of each class
class_distribution = combined_df['PCOS (Y/N)'].value_counts()

print("Combined Distribution of Diagnosis Classes:")
print(class_distribution)

# Plot the distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Diagnosis Classes')
plt.xlabel('PCOS (Y/N)')
plt.ylabel('Number of Samples')
plt.xticks(ticks=[0, 1], labels=['Not Affected (N)', 'Affected (Y)'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
