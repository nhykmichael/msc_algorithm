import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Step 1: Load the JSON data for feature importance
with open('model_3_feature_importance.json', 'r') as f: ''' CHANGE FEATURES DEPENDING ...'''
    feature_importance = json.load(f)

# Sort the features by importance and select the top N features
top_n = 20 ''' CHANGE NUMBER OF HIGHLY RELEVANT FEATURES DEPENDING ...'''
selected_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:top_n]

# Step 2: Load the mixed data
mixed_data = pd.read_csv("drebin_mix_6650.csv")          ''' CHANGE FILE DEPENDING ...'''
mixed_data['class'] = mixed_data['class'].map({'B': 0, 'S': 1})
# Ensure the mixed data has the same selected features
X_mixed = mixed_data[selected_features]

# Normalize feature data using the same scaler
scaler = MinMaxScaler()
X_mixed_scaled = scaler.fit_transform(X_mixed)
y_test_encoded = mixed_data['class']  # Ground truth labels for evaluation  ''' CHANGE LABEL DEPENDING ...'''

# Convert categorical labels to binary labels
label_encoder = LabelEncoder()
#y_test_encoded = label_encoder.fit_transform(y_test)

# Step 3: Load the pre-trained model
autoencoder = load_model('Model_3_100E.h5')             ''' CHANGE PRE-TRAINED MODEL DEPENDING ...'''

# Step 4: Evaluate the performance on mixed data
reconstruction_error_mixed = autoencoder.predict(X_mixed_scaled)

reconstruction_error_mixed = np.mean(np.square(X_mixed_scaled + reconstruction_error_mixed), axis=1)

# Calculate and print anomaly threshold
#threshold_mixed = np.mean(reconstruction_error_mixed) + 3 * np.std(reconstruction_error_mixed)
threshold_mixed = 0.009                                 ''' CHANGE CUT-OFF POINT DEPENDING ...'''
predicted_labels = (reconstruction_error_mixed > threshold_mixed).astype(int)

# Step 5: Evaluate model performance
accuracy = accuracy_score(y_test_encoded, predicted_labels)
precision = precision_score(y_test_encoded, predicted_labels)
recall = recall_score(y_test_encoded, predicted_labels)
f1 = f1_score(y_test_encoded, predicted_labels)
auc = roc_auc_score(y_test_encoded, reconstruction_error_mixed)

''' CHANGE RENAME ACCORDING TO THE MODEL ...'''

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# Step 6: Visualize results

# Plot the reconstruction error distribution
plt.figure(figsize=(6, 4))
plt.hist(reconstruction_error_mixed, bins=50, color='red', alpha=0.5)
plt.axvline(threshold_mixed, color='red', linestyle='dashed', label=f'Anomaly Threshold ={threshold_mixed}')
plt.title('Model_3 Malware Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency (2000)')
plt.legend()
plt.show()

# Plot the AUC-ROC curve
fpr, tpr, _ = roc_curve(y_test_encoded, reconstruction_error_mixed)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.title('Model_3 AUC-ROC 50-Epochs')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_encoded, predicted_labels)

# Plot the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
