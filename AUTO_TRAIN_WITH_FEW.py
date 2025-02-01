import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Load Drebin dataset
df = pd.read_csv('cleaned_drebin_class_B.csv')  # Replace with the path to your Drebin dataset

# Filter benign samples ('B' class)
benign_samples = df[df['class'] == 'B'].drop(columns=['class'])  # Drop class column

# Normalize the data (Autoencoders generally benefit from normalized input)
scaler = MinMaxScaler()
benign_scaled = scaler.fit_transform(benign_samples)

# Select top 'n' features (e.g., n=10)
n = 141  # Set n to the number of top features you want to use
selector = SelectKBest(f_classif, k=n)  # Select the top n features using ANOVA F-test
benign_selected = selector.fit_transform(benign_scaled, np.zeros(benign_scaled.shape[0]))  # Dummy target for feature selection

# Split the data into training and test sets
X_train, X_test = train_test_split(benign_selected, test_size=0.2, random_state=42)

# Build a Sparse Autoencoder model
def build_sparse_autoencoder(input_dim):
    model = Sequential()
    # Encoder
    model.add(Dense(64, activation='relu', 
                    activity_regularizer=regularizers.l1(10e-5),  # Sparsity constraint
                    input_dim=input_dim))
    # Bottleneck (compressed representation)
    model.add(Dense(32, activation='relu', 
                    activity_regularizer=regularizers.l1(10e-5)))  # Sparsity constraint
    # Decoder
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_dim, activation='sigmoid'))  # Sigmoid for normalized data
    return model

# Initialize the model
input_dim = X_train.shape[1]  # Number of selected features
autoencoder = build_sparse_autoencoder(input_dim)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
history = autoencoder.fit(X_train, X_train, 
                          epochs=100, 
                          batch_size=32, 
                          validation_data=(X_test, X_test))

# Calculate Reconstruction Error on the test data
reconstructions = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(reconstructions - X_test), axis=1)

autoencoder.save('Mode_2_50EP_141.n5')

# Set threshold as mean + 3 standard deviations
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

# Print Reconstruction Error and Threshold
print("Average Reconstruction Error on Class 'B' Test Data:", np.mean(reconstruction_error))
print("Anomaly Detection Threshold:", threshold)

# Plot training and validation loss over epochs
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model_2 Reconstruction Error 100-Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot the Reconstruction Error Distribution
plt.figure(figsize=(6, 4))
plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.title('Model_2 Error Distribution 100-Epochs')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()
