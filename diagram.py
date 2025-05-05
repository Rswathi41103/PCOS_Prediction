import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axes
fig, ax = plt.subplots(figsize=(20, 12))

# Define block positions and sizes
blocks = {
    'User Input\n(Symptoms Data)': (0.1, 0.9, 0.2, 0.08),
    'User Input\n(Clinical Data and Ultrasound Imaging Data)': (0.7, 0.9, 0.2, 0.08),
    'Preprocessing\n(Symptoms Data)': (0.1, 0.75, 0.2, 0.08),
    'Preprocessing\n(Clinical Data)': (0.7, 0.75, 0.2, 0.08),
    'Feature Extraction\n(VGG16 Model)': (0.7, 0.6, 0.2, 0.08),
    'Combining Features': (0.7, 0.45, 0.2, 0.08),
    'Model Training\n(Logistic Regression)': (0.1, 0.6, 0.2, 0.08),
    'Model Training\n(Gradient Boosting Classifier)': (0.7, 0.3, 0.2, 0.08),
    'Model Evaluation': (0.1, 0.45, 0.2, 0.08),
    'Model Evaluation': (0.7, 0.15, 0.2, 0.08),
    'Results Optimization\n(Improve Accuracy,\nTune Hyperparameters)': (0.1, 0.3, 0.2, 0.08),
    'Results Optimization\n(Improve Accuracy,\nTune Hyperparameters)': (0.7, 0.0, 0.2, 0.08),
    'Real-Time Prediction\n(Input Data,\nOutput Diagnosis)': (0.1, 0.15, 0.2, 0.08),
    'Real-Time Prediction\n(Input Data,\nOutput Diagnosis)': (0.7, -0.15, 0.2, 0.08),
    'Final Result\n(Affected/Not Affected)': (0.1, 0.0, 0.2, 0.08),
    'Final Result\n(Affected/Not Affected)': (0.7, -0.3, 0.2, 0.08)
}

# Add rectangles (blocks)
for block, (x, y, w, h) in blocks.items():
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    plt.text(x + w/2, y + h/2, block, ha='center', va='center', fontsize=10)

# Add arrows
arrows = [
    ((0.2, 0.9), (0.2, 0.83)),  # User Input to Preprocessing (Symptoms Data)
    ((0.8, 0.9), (0.8, 0.83)),  # User Input to Preprocessing (Clinical Data)
    ((0.2, 0.75), (0.2, 0.68)),  # Preprocessing (Symptoms Data) to Model Training
    ((0.8, 0.75), (0.8, 0.68)),  # Preprocessing (Clinical Data) to Feature Extraction
    ((0.8, 0.6), (0.8, 0.53)),  # Feature Extraction to Combining Features
    ((0.8, 0.45), (0.8, 0.38)),  # Combining Features to Model Training (Gradient Boosting Classifier)
    ((0.2, 0.6), (0.2, 0.53)),  # Model Training (Logistic Regression) to Model Evaluation
    ((0.2, 0.45), (0.2, 0.38)),  # Model Evaluation to Results Optimization (Symptoms)
    ((0.8, 0.3), (0.8, 0.23)),  # Model Training (Gradient Boosting) to Model Evaluation (Advanced)
    ((0.2, 0.3), (0.2, 0.23)),  # Results Optimization to Real-Time Prediction (Symptoms)
    ((0.8, 0.15), (0.8, 0.08)),  # Model Evaluation to Results Optimization (Advanced)
    ((0.2, 0.15), (0.2, 0.08)),  # Real-Time Prediction to Final Result (Symptoms)
    ((0.8, 0.0), (0.8, -0.07)),  # Results Optimization to Real-Time Prediction (Advanced)
    ((0.2, 0.0), (0.2, -0.07)),  # Real-Time Prediction to Final Result (Symptoms)
    ((0.8, -0.15), (0.8, -0.22))  # Real-Time Prediction to Final Result (Advanced)
]

for (x1, y1), (x2, y2) in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', shrink=0.05))

# Set limits and hide axes
plt.xlim(0, 1)
plt.ylim(-0.4, 1)
ax.axis('off')

# Show plot
plt.show()
