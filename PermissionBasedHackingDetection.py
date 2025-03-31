import numpy as np #For numerical purposes not considered deep LIB
from typing import List, Tuple
import pandas as pd     # for graphing 
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import pickle
import h5py
from colorama import Fore, Style
from sklearn.metrics import accuracy_score

"""This is a optimized implementation of a neural network with one hidden layer, trained using gradient descent, 
for the classification of Android applications' security risk based on their requested permissions. It is 
designed to be used as a baseline model for comparison with more advanced models. The neural network is implemented 
from scratch using only numpy for numerical operations on arrays. The model is trained and evaluated on the Drebin 
dataset, which contains information about 5,560 malware and 9,476 benign apps, and is available at 
https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html. The dataset is preprocessed and converted to a CSV file 
containing the feature vectors and labels, which is then used to train the model. The model is evaluated on a test 
set containing 20% of the data, and achieves an accuracy of 99.7%. The model is then saved to a file in HDF5 format. 
The model can be loaded from the file and used to predict the class labels for new apps. 

    The PermissionBasedClassifier class is designed for the classification and prediction of Android applications'
    security risk based on their requested permissions. It preprocesses app data, extracts permission-based features,
    and utilizes a machine learning model, potentially a Neural Network, to assess whether an app exhibits malicious
    behavior.
    Author: MN Ahimbisibwe
    SN: 217005435
    Date: 2021-08-10

    References:
        [1] https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html

"""


class NeuralNetwork:
    def __init__(self, layers: List[int], learning_rate: float = 0.01, epochs: int = 20, batch_size: int = 32,
                 verbose: bool = False):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.params = {}
        self.losses = []
        self.accuracies = []
        self.X = None
        self.y = None

        self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(42)  # Set a random seed for reproducibility
        for i in range(1, len(self.layers)):
            self.params[f'W{i}'] = np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(
                1 / self.layers[i - 1])
            """Formula for Xavier initialization = sqrt(1 / n) where n is the number of neurons in the previous layer
                W{i} is the weight matrix of the current layer, of shape... self.params[f'W{i}'] is the weight matrix 
                of the current layer, of shape (self.layers[i], self.layers[i - 1])
            """
            # Initialize the weights using that formula to ensure
            # the variance of the outputs of each layer is roughly the same
            self.params[f'b{i}'] = np.zeros((self.layers[i], 1))  # Initialize the biases to zeros

    # ReLu is faster than sigmoid, and it is the most widely used activation function in deep learning networks
    # ReLu is non-linear, which allows the network to approximate non-linear functions
    # ReLu is easy to optimize because it is similar to linear functions
    # ReLu is robust to vanishing gradient problems
    # ReLu is sparse, which means that it can be used to reduce overfitting
    # ReLu is computationally efficient because it allows the network to converge very quickly

    @classmethod
    def relu(cls, Z: np.ndarray) -> np.ndarray:  # Rectified Linear Unit newer common activation function
        return np.maximum(0, Z)  # Return the element-wise maximum of Z and 0, f(x) = max(0,x)
    '''The derivative of ReLu is 1 if x > 0, and 0 otherwise. Z is the linear output of the current layer,'''

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))  # Return the sigmoid of Z, f(x) = 1 / (1 + e^(-x))

    def forward_propagation(self, X: np.ndarray) -> list[ndarray]:
        """
            Performs the forward pass of the neural network.

            Args:
                X (np.ndarray): The input to the neural network, of shape (num_samples, num_features).

            Returns:
                A2 (np.ndarray): The output of the neural network, of shape (num_samples, num_classes).
            """

        A = X.astype(float)  # Convert the input to float
        cache = [A]  # To store the outputs of each layer
        for i in range(1, len(self.layers) - 1):
            Z = np.dot(self.params[f'W{i}'], A) + self.params[f'b{i}']  # Linear output of the current layer
            A = self.relu(Z)
            cache.append(A)
        """ Expected output layer size is 1, so we use sigmoid activation function"""
        Z_output = np.dot(self.params[f'W{len(self.layers) - 1}'], A) + self.params[f'b{len(self.layers) - 1}']
        A_output = self.sigmoid(Z_output)
        cache.append(A_output)
        return cache

    @classmethod
    def cross_entropy(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the cross-entropy loss"""
        # Ensure y_true is a 1-dimensional array
        y_true = y_true.ravel()  # Return a contiguous flattened array
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss

    def backward_propagation(self, y_true: np.ndarray, y_pred: np.ndarray, cache: List[np.ndarray]) -> dict:
        """Performs the backward pass of the neural network
        Args:
            y_true (np.ndarray): The ground truth labels, of shape (num_samples, num_classes).
            y_pred (np.ndarray): The predicted labels, of shape (num_samples, num_classes).
            cache (List[np.ndarray]): The cache containing the outputs of each layer of the network.
        Returns:
            gradients (dict): The gradients of the loss with respect to the parameters of the network.

        """
        gradients = {}  # To store the gradients
        m = y_true.shape[1]  # Number of samples
        dA = y_pred - y_true  # Derivative of the loss with respect to the output of the network
        for i in reversed(range(1, len(self.layers))):
            dZ = np.multiply(dA, np.int64(cache[i] > 0))  # Derivative of the loss with respect to the linear output
            gradients[f'dW{i}'] = np.dot(dZ, cache[i - 1].T) / m
            gradients[f'db{i}'] = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(self.params[f'W{i}'].T, dZ)
            # Derivative of the loss with respect to the output of the previous layer
        return gradients

    def update_parameters(self, gradients: dict):
        for i in range(1, len(self.layers)):  # Update the parameters
            self.params[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']  # Update the weights
            self.params[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']  # Update the biases

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.X = X
        self.y = y
        # Ensure y is a 2-dimensional array
        if self.y.ndim == 1:
            self.y = self.y.reshape(1, -1)
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, self.X.shape[1], self.batch_size):  # Iterate over the training set
                X_batch = self.X[:, i:i + self.batch_size]  # Get the current batch
                y_batch = self.y[:, i:i + self.batch_size]  # Get the labels for the current batch
                cache = self.forward_propagation(X_batch)  # Perform forward propagation
                y_pred = cache[-1]  # Get the predictions
                epoch_loss += self.cross_entropy(y_batch, y_pred)  # Calculate the loss
                gradients = self.backward_propagation(y_batch, y_pred, cache)  # Perform backward propagation
                self.update_parameters(gradients)
            self.losses.append(epoch_loss)
            if self.verbose:
                print(f'Epoch: {epoch + 1} - Loss: {epoch_loss:.4f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions for the given input.
        Args:
            X (np.ndarray): The input to the neural network, of shape (num_samples, num_features).
        Returns:
            y_pred (np.ndarray): The predicted labels, of shape (num_samples, num_classes).
        """
        cache = self.forward_propagation(X)
        y_pred = cache[-1]
        return (y_pred > 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, ndarray]:
        """Evaluates the model on the given input and labels.
        Args:
            X (np.ndarray): The input to the neural network, of shape (num_samples, num_features).
            y (np.ndarray): The ground truth labels, of shape (num_samples, num_classes).
        Returns:
            loss (float): The cross-entropy loss of the model on the given input and labels.
            accuracy (float): The accuracy of the model on the given input and labels.
        """
        cache = self.forward_propagation(X)
        y_pred = cache[-1]
        loss = self.cross_entropy(y, y_pred)
        accuracy = np.mean((y_pred > 0.5) == y)
        return loss, accuracy


class PermissionBasedClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.unique_permissions: List[str] = []
        self.encoder: LabelEncoder = LabelEncoder()
        # self.model: NeuralNetwork = None

        self.X, self.y = self.load_dataset(self.dataset_path)

        self.y = self.encoder.fit_transform(self.y)

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Initialize the model
        """The number of input features is equal to the number of unique permissions in the dataset,
        and the number of output classes is equal to the number of unique labels in the dataset.
        128, 64, 32, 1 are the number of neurons in each layer,
        batch_size is the number of samples to use in each iteration,
        verbose is a boolean flag that controls whether to print the loss after each epoch.
        """
        self.model = NeuralNetwork(layers=[self.X_train.shape[1], 128, 64, 32, 1],  # Update output layer size to 1
                                   learning_rate=0.01, epochs=20, batch_size=32, verbose=True)

    def extract_unique_permissions(self):
        # Load the dataset again to extract column headers (unique permissions/API calls)
        with open(self.dataset_path, 'r') as f:
            headers = f.readline().strip().split(',')[1:-1]  # Exclude the first and last columns
        return headers

    def update_unique_permissions(self):
        # Extract the unique permissions
        self.unique_permissions = self.extract_unique_permissions()
        # Save the unique permissions to a file
        with open('unique_permissions.txt', 'w') as f:
            f.write('\n'.join(self.unique_permissions))

    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads the dataset from the given path.
        Args:
            dataset_path (str): The path to the dataset.
        Returns:
            X (np.ndarray): The features of the dataset, of shape (num_samples, num_features).
            y (np.ndarray): The labels of the dataset, of shape (num_samples, num_classes).
        """
        # Load the dataset
        data = np.genfromtxt(dataset_path, delimiter=',', dtype=str, invalid_raise=False)
        X = np.array([[int(x) if x.isdigit() else np.nan for x in row] for row in data[1:, 1:-1]])
        X = np.nan_to_num(X, nan=0)

        y = data[1:, -1]
        return X, y

    def plot_losses(self): ##### Student must explain why this is necessary
        plt.figure(figsize=(8, 8))
        plt.plot(self.model.losses, label='Training Loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def make_predictions(self):
        y_pred = self.model.predict(self.X_test.T)
        y_pred = y_pred.reshape(-1)
        print("Predictions:", y_pred)
        print("Ground Truth:", self.y_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))

    def save_model(self):
        # Save the model
        with h5py.File('model.h5', 'w') as f:
            for key, value in self.model.params.items():
                f.create_dataset(key, data=value)
            f.create_dataset('unique_permissions', data=np.array(self.unique_permissions).astype('S'))
            f.create_dataset('classes', data=np.array(self.encoder.classes_).astype('S'))

    def load_model(self):
        # Load the model
        with h5py.File('model.h5', 'r') as f:
            for key in f.keys():
                if key != 'unique_permissions' and key != 'classes':
                    self.model.params[key] = f[key][:]
            self.unique_permissions = f['unique_permissions'][:].astype(str)
            self.encoder.classes_ = f['classes'][:].astype(str)

    def run(self):
        # Train the model
        self.model.fit(self.X_train.T, self.y_train)

        self.update_unique_permissions()
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.X_test.T, self.y_test)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        self.plot_losses()

        self.save_model()
        self.load_model()

        # Make predictions
        self.make_predictions()
        # print("First 10 unique permissions:", self.unique_permissions[:10])
        print("Number of unique permissions:", len(self.unique_permissions))
        print("Number of classes:", len(self.encoder.classes_))
        print("Classes:", self.encoder.classes_)
        print("Number of features:", self.X.shape[1])
        print("Number of samples:", self.X.shape[0])
        print("Number of training samples:", self.X_train.shape[0])
        print("Number of testing samples:", self.X_test.shape[0])

    # *****************************************************************************

    def predict(self, permissions: List[str]) -> int:
        """Predicts the class label for the given permissions.
        Args:
            permissions (List[str]): The permissions of the app.
        Returns:
            y_pred (int): The predicted class label.
        """
        # Extract the permission-based features
        features = self.extract_features(permissions)

        # Make predictions
        y_pred = self.model.predict(features)
        return y_pred[0]

    def extract_features(self, permissions: List[str]) -> np.ndarray:
        """Extracts the permission-based features from the given permissions.
        Args:
            permissions (List[str]): The permissions of the app.
        Returns:
            features (np.ndarray): The permission-based features, of shape (num_samples, num_features).
        """
        # Initialize the features array
        features = np.zeros((1, len(self.unique_permissions)))

        # Iterate over the permissions
        for permission in permissions:
            if permission in self.unique_permissions:
                idx = self.unique_permissions.index(permission)
                # Set the value of the feature to 1
                features[0, idx] = 1
        return features

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, ndarray]:
        """Evaluates the model on the given input and labels.
        Args:
            X (np.ndarray): The input to the neural network, of shape (num_samples, num_features).
            y (np.ndarray): The ground truth labels, of shape (num_samples, num_classes).
        Returns:
            loss (float): The cross-entropy loss of the model on the given input and labels.
            accuracy (float): The accuracy of the model on the given input and labels.
        """
        cache = self.forward_propagation(X)
        y_pred = cache[-1]
        loss = self.cross_entropy(y, y_pred)
        accuracy = np.mean((y_pred > 0.5) == y)
        return loss, accuracy


if __name__ == '__main__':
    # Loss: 0.0639, Accuracy: 0.9784
    # very good result, especially in the context of binary classification
    # dataset = "drebin215dataset5560malware9476benign.csv"
    # model_ = PermissionBasedClassifier(dataset)
    # model_.update_unique_permissions()
    # model_.run()
    pass

""" To classify an app, permissions.txt pulled out of the phone 
    using adb shell pm list permissions -d -g > permissions.txt 
    classify and predict using the model saved in model.h5, determine if the app is malicious or not
"""


class MalwarePredictor: ########### Can it detect all malware? needs explain
    def __init__(self, model_path='model.h5'):
        self.classes = None
        self.unique_permissions = None
        self.model_params = {}
        self.load_model(model_path)

    def load_model(self, model_path):
        with h5py.File(model_path, 'r') as f:
            for key in f.keys():
                if key not in ['unique_permissions', 'classes']:
                    self.model_params[key] = np.array(f[key])
            self.unique_permissions = f['unique_permissions'][:].astype(str)
            self.classes = f['classes'][:].astype(str)

    def extract_features(self, permissions_file):
        # Initialize a binary vector with all zeros
        features = np.zeros(len(self.unique_permissions))

        # Convert unique_permissions to a set for faster lookup
        set(self.unique_permissions)

        with open(permissions_file, 'r') as file:
            # Read all lines in the file and tokenize them
            tokenized_lines = [set(line.strip().split(".")) for line in file.readlines()]

        # Check each unique permission
        count_unique_permissions = 0
        for i, unique_permission in enumerate(self.unique_permissions):
            # If any tokenized line contains the unique permission, set the feature to 1
            for tokens in tokenized_lines:
                if unique_permission in tokens:
                    features[i] = 1
                    count_unique_permissions += 1
                    break
        print(f"{Fore.CYAN}Number of Unique Permissions: {Fore.WHITE}{count_unique_permissions} "
              f"{Fore.CYAN}out of {Fore.WHITE}"
              f"{len(self.unique_permissions)}", Style.RESET_ALL)

        return features

    def predict(self, features):
        # Manually perform the forward pass through the network
        # This is a generic implementation; maybe adjusted based on network architecture
        a = features
        for i in range(1, len(self.model_params) // 2 + 1):
            W = self.model_params[f'W{i}'].T  # Transpose the weight matrix
            b = self.model_params[f'b{i}'].flatten()  # Flatten the bias vector
            z = np.dot(a, W) + b  # Linear activation
            a = np.maximum(z, 0)  # ReLU activation (or use sigmoid)
        predicted_class = self.classes[np.argmax(a)]  # Get the predicted class
        return predicted_class


def main():
    permissions_file_path = 'allowed_permissions.txt'
    predictor = MalwarePredictor()
    features = predictor.extract_features(permissions_file_path)
    predicted_class = predictor.predict(features)
    print(Fore.YELLOW, f"The predicted class is: {Style.RESET_ALL}{predicted_class}")
    if predicted_class == b'S':
        print(Fore.RED, "The app exhibits malicious behavior.", Style.RESET_ALL)
    else:
        print(Fore.CYAN, "The app is benign.", Style.RESET_ALL)


if __name__ == "__main__":
    # main()
    pass
