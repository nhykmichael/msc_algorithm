import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load extracted features
with open("Model_6.json", "r") as f:
    feature_importance = json.load(f)
#selected_features = feature_data["features"]

# Select top 20 features by importance
sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
top_N_features = [feature[0] for feature in sorted_features[:40]]

# Load benign-only dataset
file_path = "real_legitimate_v1.csv"  # Replace with your file path
data = pd.read_csv(file_path)
# Ensure only selected features are used
data = data.drop(columns=['Package', 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
data = data.iloc[:,:-11]

#data.fillna(method='ffill', inplace=True)
X = data[top_N_features]

# Handle missing values (if any)
for col in X.columns:
    # Ensure the column is of numeric type
    if pd.api.types.is_numeric_dtype(X[col]):
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)
    else:
        print(f"Column '{col}' is not numeric and will be ignored for mean imputation.")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and validation sets
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Define Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
model = SparseAutoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)

# Training parameters
epochs = 50
batch_size = 256
train_loss_history = []
val_loss_history = []

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i in range(0, X_train_tensor.size(0), batch_size):
        batch = X_train_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= (X_train_tensor.size(0) // batch_size)
    train_loss_history.append(train_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, X_val_tensor).item()
        val_loss_history.append(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "Model_WORST_50EP.pth")
print("Model saved as Model_6_50EP.pth")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_loss_history, label="Training Loss")
plt.plot(range(1, epochs + 1), val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# Reconstruction error distribution
model.eval()
with torch.no_grad():
    reconstructions = model(X_train_tensor)
    reconstruction_error = ((X_train_tensor - reconstructions) ** 2).mean(dim=1).numpy()
    threshold = np.mean(reconstruction_error + 3 * np.std(reconstruction_error))

plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.75, color="blue")
plt.axvline(threshold, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.savefig("reconstruction_error_distribution.png")
plt.show()
