import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Step 1: Load the full dataset for feature extraction using TabNet
full_data = pd.read_csv("drebin.csv")
target_variable = "class"  # Define target column

# Encode target variable
label_encoder = LabelEncoder()
full_data[target_variable] = label_encoder.fit_transform(full_data[target_variable])

# Separate features and target
X = full_data.drop(columns=[target_variable])
y = full_data[target_variable]

# Normalize feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_scaled = imputer.fit_transform(X_scaled)

# Step 2: Train TabNet for feature importance extraction
tabnet = TabNetClassifier()
tabnet.fit(
    X_scaled, y,
    max_epochs=50,
    patience=5,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Extract feature importance from TabNet
feature_importances = tabnet.feature_importances_
features = X.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Convert DataFrame to dictionary 
importance_dict = importance_df.set_index('Feature').to_dict()['Importance'] 
# Save the dictionary to a JSON file 
with open('model_6_feature_importance.json', 'w') as json_file: 
    json.dump(importance_dict, json_file, indent=4)


# Sort the features by importance and select the top 20

# Load feature importance data from JSON
with open('model_6_feature_importance.json', 'r') as f:
    extracted_features = json.load(f)

# Convert the dictionary to a DataFrame
features_df = pd.DataFrame(list(extracted_features.items()), columns=['Feature', 'Importance'])

# Sort the DataFrame by importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False)

# Select the top N features
top_n = 20
selected_features = features_df.head(top_n)['Feature'].tolist()

# Save selected features for later use
# with open('selected_features_Model_3.json', 'w') as f:
     # json.dump(selected_features, f)

# Step 3: Load the cleaned class 'B' data and filter by selected features
X_class_b = pd.read_csv("cleaned_drebin_class_B.csv")
X_class_b_selected = X_class_b[selected_features]

# Step 4: Build and train the Sparse Autoencoder on selected features from class 'B' data
input_dim = X_class_b_selected.shape[1]
autoencoder = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,), activity_regularizer=regularizers.l1(1e-5)),
    Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(32, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(128, activation='relu', activity_regularizer=regularizers.l1(1e-5)),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the Sparse Autoencoder on class 'B' data with selected features
history = autoencoder.fit(
    X_class_b_selected, X_class_b_selected,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

autoencoder.save('Model_6_50E.h5')
#print("Model saved as 'Model_3_50E.h5'")

# Evaluate Model Performance on Reconstruction Error
reconstruction_error = autoencoder.predict(X_class_b_selected)
reconstruction_error = np.mean(np.square(X_class_b_selected - reconstruction_error), axis=1)

# Calculate and print anomaly threshold
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

print("Reconstruction Error on Class 'B' Data:", np.mean(reconstruction_error))
print("Anomaly Threshold:", threshold)

#Plot the top 20 features 
top_20_features = importance_df.head(20) 
plt.figure(figsize=(7, 4)) 
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis') 
plt.title('Top 20 Features by Importance (TabNet)') 
plt.xlabel('Feature Importance') 
plt.ylabel('Features') 
plt.tight_layout() 
plt.show()


# Plot the Training and Validation Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model_3 Reconstruction Error 50-Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot the Reconstruction Error Distribution
plt.figure(figsize=(6, 4))
plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.title('Model_3 Error Distribution 100-Epochs')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# #Plot the top 20 features 
# top_20_features = importance_df.head(20) 
# plt.figure(figsize=(7, 4)) 
# sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis') 
# plt.title('Top 20 Features by Importance (TabNet)') 
# plt.xlabel('Feature Importance') 
# plt.ylabel('Features') 
# plt.tight_layout() 
# plt.show()
