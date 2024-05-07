import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Generate synthetic numerical data for demonstration
num_samples = 1000
num_features = 50
X = np.random.randn(num_samples, num_features)  # Input features
y = np.random.randint(2, size=num_samples)      # Binary labels (0: non-pedestrian, 1: pedestrian)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy}')

# Make predictions on a sample test data
sample_index = 0
sample_data = X_test[sample_index].reshape(1, -1)
prediction = model.predict(sample_data)
print(f'Prediction for sample {sample_index}: {prediction[0]}')
sample_index = 0
sample_data = X_test[sample_index].reshape(1, -1)
prediction = model.predict(sample_data)
print(f'Sample Data: {sample_data}')
print(f'Prediction for sample {sample_index}: {prediction[0]}')
