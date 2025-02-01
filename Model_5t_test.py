import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import load_model
import itertools

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Load and preprocess the dataset
def data_proc_with_labels(file_name):
    df = pd.read_csv(file_name)
    df_labels = df['Malware']
    #df = df.drop(columns=['Family', 'Hash'])
    #df = df.iloc[:, :-3]
    df = df.drop(columns=['Package', 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
    df = df.iloc[:,:139]
    return df , df_labels

# Normalize the data
def normalise(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

# Load dataset
data = 'all.csv'
df, true_labels = data_proc_with_labels(data)
df_scaled = normalise(df)

# Load the pre-trained autoencoder model
autoencoder = load_model('Model_5_50EP.h5')

# Predict the reconstructed output on the test data
X_test_pred = autoencoder.predict(df_scaled)

# Calculate the reconstruction error (MSE)
MSE = np.mean(np.square(df_scaled - X_test_pred), axis=1)

# Determine the threshold from the pre-trained model
threshold = 0.003365574325347673

# Classify as anomalous or not based on the threshold
anomaly = MSE > threshold
predicted_labels = anomaly.astype(int)


# Evaluate the model
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Model 5t 50-Epoch')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign', 'Malicious'], rotation=45)
plt.yticks(tick_marks, ['Benign', 'Malicious'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# Plot the Reconstruction Error Distribution for Malware Samples
# Separate MSE by true labels
mse_benign = MSE[true_labels == 0]  # Benign samples
mse_malware = MSE[true_labels == 1]  # Malware samples

plt.figure(figsize=(6, 4))
#plt.hist(MSE, bins=50, color='blue', alpha=0.7)
plt.hist(mse_benign, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='black', linestyle='dashed', label=f'Anomaly Threshold')
plt.title('Model_5t Error Distribution 50-Epoch')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(mse_malware, bins=50, color='red', alpha=0.7, label='Malicious')
plt.axvline(threshold, color='black', linestyle='dashed', label=f'Anomaly Threshold={threshold:.4f}')
#plt.axvline(threshold, color='purple', linestyle='dashed', label='Anomaly Threshold')
plt.title('Model_5t Malware Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
#plt.hist(MSE, bins=50, color='blue', alpha=0.7)
plt.hist(mse_benign, bins=50, color='blue', alpha=0.7, label='Benign')
plt.hist(mse_malware, bins=50, color='red', alpha=0.7, label='Malicious')
plt.axvline(threshold, color='black', linestyle='dashed', label=f'Anomaly Threshold={threshold:.4f}')
#plt.axvline(threshold, color='purple', linestyle='dashed', label='Anomaly Threshold')
plt.title('Model_5t B/S Error Distribution 50-Epoch')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(true_labels, MSE)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model_5t AUC(ROC) 50-Epochs')
plt.legend(loc="lower right")
plt.show()
