import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset and drop irrelevant columns

def data_proc(file_name):
    df = pd.read_csv(file_name)
       
    #df = df.iloc[:, :-11] # Start=0, sha256 @ 19
    # Fill with a specific value
    #df.fillna(0, inplace=True)

    # Fill with mean of the column
    #df.fillna(df.mean(), inplace=True)

    # Fill with median of the column
    #df.fillna(df.median(), inplace=True)

    # Fill with mode of the column
    #df.fillna(df.mode().iloc[0], inplace=True)

    #df = df.drop(columns=['Malware'])
    df = df.drop(columns=['Package', 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
    df = df.iloc[:,:-11]
    return df

# Normalize the data
def normalise(df):
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    df_scaled = scaler.fit_transform(df)  # Use the scaler to normalize the data
    return df_scaled

# Split the data into training and test sets
def training_data_split(df_scaled, test_size=0.2, random_state=42):
    X_train, X_test = train_test_split(df_scaled, test_size=test_size, random_state=random_state)
    return X_train, X_test

# Build a sparse autoencoder model
def build_sparse_autoencoder(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', activity_regularizer=regularizers.l1(10e-5), input_dim=input_dim))
    model.add(Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_dim, activation='sigmoid'))  # Sigmoid for normalized data
    return model

# Initialize the model, compile, and train
def train(file_name):
    df = data_proc(file_name)
    df_scaled = normalise(df)
    X_train, X_test = training_data_split(df_scaled)
    input_dim = X_train.shape[1]
    autoencoder = build_sparse_autoencoder(input_dim)
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')
    # Train the model
    history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))
    #history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)
    
    # Predict the reconstructed output on the test data
    X_test_pred = autoencoder.predict(X_test)
    
    # Calculate the reconstruction error (MSE) on the test set
    MSE = np.mean(np.square(X_test - X_test_pred), axis=1)  # MSE between original and reconstructed data
    #autoencoder.save('Model_5_100EP.h5')
    
    # Calculate mean and standard deviation of reconstruction error
    mean_error = np.mean(MSE)
    std_error = np.std(MSE)
    # Set threshold as mean + 3 standard deviations
    threshold = 0.003365574325347673
    
    print(f'Reconstruction Error (MSE): {mean_error}')    
    print(f"Anomaly detection threshold: {threshold}")
    
    # Plot training and validation loss over epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    #plt.title('Model 5 Training/Validating (Loss)')
    plt.title('Model_5 Reconstruction Error 50-Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()
    
    plt.hist(MSE, bins=50, color='blue', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', label='Anomaly Threshold')
    plt.title('Model_1 Error Distribution 50-Epochs')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    

# Entry point
if __name__ == '__main__':
    data = 'real_legitimate_v1.csv' 
    train(data)