from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Set paths to your dataset
dataset_path = 'D:\\Periods\\PCOS'

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False)

# Define a new model that outputs features from the last convolutional block
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Set up the data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Generate batches of images and labels
batch_size = 32
img_height, img_width = 224, 224

generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Extract features
num_images = generator.samples
features = model.predict(generator, steps=num_images // batch_size + 1)

# Check the shape of the extracted features
print(f"Shape of extracted features: {features.shape}")

# Reshape features
image_features = features.reshape((num_images, -1))
image_labels = generator.classes

# Check the shape after reshaping
print(f"Shape of reshaped features: {image_features.shape}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(image_features, image_labels, test_size=0.2, random_state=42)

# Train a classifier on the extracted features
classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

# Save features and labels for later use
np.save('image_features.npy', image_features)
np.save('image_labels.npy', image_labels)
