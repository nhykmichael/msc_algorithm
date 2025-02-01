import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import json
from tensorflow.keras import regularizers

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

with open('Model_4_selected_features.json', 'r') as f:
    selected_features = json.load(f)
    
sorted_features = sorted(selected_features, key=lambda x: x[1], reverse=True)
top_N_features = [feature[0] for feature in sorted_features[:40]]
    
# Step 2: Load the cleaned class 'B' data and filter by selected top features
X_class_b = pd.read_csv("real_legitimate_v1.csv")
X_class_b = X_class_b.drop(columns=['Package', "Malware", 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
X_class_b = X_class_b.iloc[:,:-11]
X_class_b_selected = X_class_b[top_N_features]

# Normalize the input data
scaler = StandardScaler()
X_class_b_selected = scaler.fit_transform(X_class_b_selected)

# Step 3: Build and train the Autoencoder on selected features from class 'B' data
input_dim = X_class_b_selected.shape[1]
autoencoder = Sequential([
    Dense(256, activation='relu', input_shape=(input_dim,), activity_regularizer=regularizers.l1(1e-5)),
    Dense(128, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(32, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(128, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(256, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder on class 'B' data with selected features
history = autoencoder.fit(X_class_b_selected, X_class_b_selected, epochs=50, batch_size=256, validation_split=0.2, verbose=1)

#autoencoder.save('Model_4_50E.h5')

# Evaluate Model Performance on Reconstruction Error
reconstruction_error = autoencoder.predict(X_class_b_selected)
reconstruction_error = np.mean(np.square(X_class_b_selected - reconstruction_error), axis=1)

# Calculate and print anomaly threshold
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

print("Reconstruction Error on Class 'B' Data:", np.mean(reconstruction_error))
print("Anomaly Threshold:", threshold)

# Plot the Training and Validation Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model_4 Reconstruction Error 50-Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot the Reconstruction Error Distribution
plt.figure(figsize=(6, 4))
plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.title('Model_4 Error Distribution 50-Epochs')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()
