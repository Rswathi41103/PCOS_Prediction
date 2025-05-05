import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load your dataset
df = pd.read_csv('PCOS_data.csv')

# Define required columns for training
required_columns = ['PCOS (Y/N)', 'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI',
                    'Cycle(R/I)', 'Pulse rate(bpm)', 'RR (breaths/min)', 'Pregnant(Y/N)',
                    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
                    'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']

# Check if all required columns are present
print("Columns in the dataset:")
print(df.columns)

# Subset the dataframe with only the required columns
df = df[required_columns]

# Print columns after subsetting
print("Columns after subsetting:")
print(df.columns)

# Replace values in 'PCOS (Y/N)' and binary columns
df["PCOS (Y/N)"] = df["PCOS (Y/N)"].replace({'Y': 1, 'N': 0})
binary_columns = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
                  'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'Cycle(R/I)']
df[binary_columns] = df[binary_columns].replace({'Y': 1, 'N': 0, 'R': 0, 'I': 1})

# Replace specific values in 'Cycle(R/I)' column
df['Cycle(R/I)'] = df['Cycle(R/I)'].replace({2: 0, 4: 1, 5: 0})

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Print the shape of the dataframe
print("Shape of the dataframe after preprocessing:")
print(df.shape)

# Extract features and labels
features = df.drop(columns=['PCOS (Y/N)']).values
labels = df['PCOS (Y/N)'].values

# Print the shape of features and labels
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Save the features and labels to numpy files (optional)
np.save('basic_data_features.npy', features)
np.save('basic_data_labels.npy', labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Print the shape of the train and test sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=10000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print test accuracy and classification report
print(f"Test Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained model
model_filename = 'basic_logistic_model.pkl'
joblib.dump(model, model_filename)

print(f"Model saved as {model_filename}")


