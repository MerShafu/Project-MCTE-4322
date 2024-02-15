import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your dataset
file_path = "HeartAttackDataset2.xlsx"
data = pd.read_excel(file_path)

# Debugging: Print the number of rows and columns in the loaded dataset
print(f"Number of rows and columns in the loaded dataset: {data.shape}")

# Assuming your data has columns 'input1', 'input2', ..., 'input7', and 'output'
X = data[['Age', 'Cholesterol', 'BPS', 'Smoking', 'Diabetes', 'BMI', 'Physical Activity Days Per Week']].values
y = data['Heart Attack Risk'].values

# Debugging: Print the number of rows in X and y
print(f"Number of rows in X: {len(X)}")
print(f"Number of rows in y: {len(y)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    #tf.keras.layers.Dense(512, activation='relu', input_shape=(7,)),
    #tf.keras.layers.Dense(256, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)

# Plot the training loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save the trained model
model.save("heart_attack_prediction_model.h5")

print(f"Accuracy on the test set: {accuracy}")