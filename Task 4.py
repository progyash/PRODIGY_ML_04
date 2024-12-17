import os
import numpy as np
try:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow'.")
    raise

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Dataset Directory (adjust this to your local directory)
DATASET_DIR = r'C:\Users\kisho\Downloads\archive (1)\leapGestRecog\leapGestRecog'  # Update this to your dataset directory
IMG_HEIGHT = 64
IMG_WIDTH = 64

def load_data(dataset_dir, img_height, img_width):
    images = []
    labels = []
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
                img_path = os.path.join(root, file)
                # Load image and preprocess
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                images.append(img_array)
                
                # Extract label from the folder structure
                label = os.path.basename(root)  # Get the name of the current folder as label
                labels.append(label)

    return np.array(images), np.array(labels)

# Load dataset
print("Loading dataset...")
X, y = load_data(DATASET_DIR, IMG_HEIGHT, IMG_WIDTH)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected (Dense) layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Add Dropout for regularization
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 10
BATCH_SIZE = 32

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Visualize training results
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
