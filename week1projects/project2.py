##2Build and train a simple neural network using a framework like TensorFlow or PyTorch

import tensorflow as tf
import os
import numpy as np

# Define paths to your training, validation, and testing data
train_dir = "/home/rubeena-sulthana/Downloads/train"
test_dir = "/home/rubeena-sulthana/Downloads/test"
val_dir = "/home/rubeena-sulthana/Downloads/validation"

# Initialize perceptron parameters
learning_rate = 0.01
num_epochs = 10
input_shape = (28, 28, 3)  # Assuming images are resized to 28x28 and have 3 channels

# Function to load and preprocess data using TensorFlow
def load_data_tf(folder_path, input_shape):
    X = []
    y = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in JPG or PNG format
                img_path = os.path.join(root, file)
                label = 1 if "positive" in root else 0  # Example: folder structure decides the label
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                X.append(img_array)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Load training data
X_train, y_train = load_data_tf(train_dir, input_shape)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0

# Define the neural network architecture using TensorFlow's Sequential API
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so using sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary crossentropy for binary classification
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=num_epochs)

# Function to evaluate accuracy on a dataset
def evaluate_accuracy(X, y):
    X = X / 255.0  # Normalize pixel values to [0, 1]
    _, accuracy = model.evaluate(X, y)
    return accuracy

# Evaluate training accuracy
train_accuracy = evaluate_accuracy(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Load validation data
X_val, y_val = load_data_tf(val_dir, input_shape)

# Evaluate validation accuracy
validation_accuracy = evaluate_accuracy(X_val, y_val)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# Load testing data
X_test, y_test = load_data_tf(test_dir, input_shape)

# Evaluate test accuracy
test_accuracy = evaluate_accuracy(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

