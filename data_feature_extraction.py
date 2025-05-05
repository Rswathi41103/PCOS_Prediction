import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed CSV file
data_csv_path = 'PCOS_data.csv'
df = pd.read_csv(data_csv_path)

# Calculate Follicle_count
df['Follicle_count'] = df['Follicle No. (L)'] + df['Follicle No. (R)']

# Replace values in Cycle(R/I)
df["Cycle(R/I)"] = df["Cycle(R/I)"].replace({2: 0, 4: 1, 5: 0})

# Extract relevant features
numerical_features = df[['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Pulse rate(bpm)',
                         'RR (breaths/min)', 'No. of aborptions', 'FSH(mIU/mL)',
                         'TSH (mIU/L)', 'LH(mIU/mL)', 'AMH(ng/mL)', 'PRL(ng/mL)',
                         'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'FSH/LH',
                         'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Waist:Hip Ratio',
                         'Follicle_count']].values

categorical_features = df[['Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
                           'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
                           'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'Cycle(R/I)']].values

# Combine numerical and categorical features
data_features = np.concatenate((numerical_features, categorical_features), axis=1)

# Extract labels
labels = df['PCOS (Y/N)'].values

# Save extracted features and labels to disk
np.save('data_features.npy', data_features)
np.save('data_labels.npy', labels)

print("Data features and labels extraction complete. Saved to disk.")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, random_state=42)

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
