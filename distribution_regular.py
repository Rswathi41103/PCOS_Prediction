import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data_csv_path = 'PCOS_data.csv'
df = pd.read_csv(data_csv_path)

# Replace values in 'PCOS (Y/N)' and binary columns if needed
df["PCOS (Y/N)"] = df["PCOS (Y/N)"].replace({'Y': 1, 'N': 0})

# Count the occurrences of each class in the 'PCOS (Y/N)' column
class_distribution = df['PCOS (Y/N)'].value_counts()

# Plot the distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Diagnosis Classes')
plt.xlabel('PCOS (Y/N)')
plt.ylabel('Number of Samples')
plt.xticks(ticks=[0, 1], labels=['Not Affected (N)', 'Affected (Y)'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
