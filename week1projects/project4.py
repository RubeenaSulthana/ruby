##4Experiment with different regularization techniques on a neural network
import tensorflow as tf
import numpy as np
import os

# Assuming OpenCV is installed for image processing
train_dir = "/home/rubeena-sulthana/Downloads/train"
test_dir = "/home/rubeena-sulthana/Downloads/test"
val_dir = "/home/rubeena-sulthana/Downloads/validation"

# Initialize parameters
learning_rate = 0.01
num_epochs = 10
input_shape = (28, 28, 3)  # Assuming images are resized to 28x28 and have 3 channels

# Regularization parameters
l2_regularization = 0.001  # L2 regularization parameter
dropout_rate = 0.2  # Dropout rate
weight_decay = 0.0001  # Weight decay parameter

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
X_train = X_train / 255.0

# Load testing data
X_test, y_test = load_data_tf(test_dir, input_shape)
X_test = X_test / 255.0

# Load validation data
X_val, y_val = load_data_tf(val_dir, input_shape)
X_val = X_val / 255.0

# Define the neural network architecture for L2 regularization
model_l2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
])

# Compile the model for L2 regularization
model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with L2 regularization
history_l2 = model_l2.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=1)

# Evaluate accuracy for L2 regularization on train data
l2_train_accuracy = model_l2.evaluate(X_train, y_train, verbose=0)[1]
print(f"L2 Regularization Train Accuracy: {l2_train_accuracy * 100:.2f}%")

# Evaluate accuracy for L2 regularization on validation data
l2_val_accuracy = model_l2.evaluate(X_val, y_val, verbose=0)[1]
print(f"L2 Regularization Validation Accuracy: {l2_val_accuracy * 100:.2f}%")

# Evaluate accuracy for L2 regularization on test data
l2_test_accuracy = model_l2.evaluate(X_test, y_test, verbose=0)[1]
print(f"L2 Regularization Test Accuracy: {l2_test_accuracy * 100:.2f}%")

# Define the neural network architecture for Dropout regularization
model_dropout = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model for Dropout regularization
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with Dropout regularization
history_dropout = model_dropout.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=1)

# Evaluate accuracy for Dropout regularization on train data
dropout_train_accuracy = model_dropout.evaluate(X_train, y_train, verbose=0)[1]
print(f"Dropout Train Accuracy: {dropout_train_accuracy * 100:.2f}%")

# Evaluate accuracy for Dropout regularization on validation data
dropout_val_accuracy = model_dropout.evaluate(X_val, y_val, verbose=0)[1]
print(f"Dropout Validation Accuracy: {dropout_val_accuracy * 100:.2f}%")

# Evaluate accuracy for Dropout regularization on test data
dropout_test_accuracy = model_dropout.evaluate(X_test, y_test, verbose=0)[1]
print(f"Dropout Test Accuracy: {dropout_test_accuracy * 100:.2f}%")

# Define the neural network architecture for Weight Decay regularization
model_weight_decay = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
])

# Compile the model for Weight Decay regularization
optimizer_weight_decay = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model_weight_decay.compile(optimizer=optimizer_weight_decay, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with Weight Decay regularization
history_weight_decay = model_weight_decay.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=1)

# Evaluate accuracy for Weight Decay regularization on train data
weight_decay_train_accuracy = model_weight_decay.evaluate(X_train, y_train, verbose=0)[1]
print(f"Weight Decay Train Accuracy: {weight_decay_train_accuracy * 100:.2f}%")

# Evaluate accuracy for Weight Decay regularization on validation data
weight_decay_val_accuracy = model_weight_decay.evaluate(X_val, y_val, verbose=0)[1]
print(f"Weight Decay Validation Accuracy: {weight_decay_val_accuracy * 100:.2f}%")

# Evaluate accuracy for Weight Decay regularization on test data
weight_decay_test_accuracy = model_weight_decay.evaluate(X_test, y_test, verbose=0)[1]
print(f"Weight Decay Test Accuracy: {weight_decay_test_accuracy * 100:.2f}%")