import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

def preprocess_data(input_data):
    """
    Preprocesses the Drebin dataset by filtering for benign ('B') instances and collecting class distribution metrics.

    Args:
        input_data (pandas.DataFrame): The input Drebin dataset.

    Returns:
        pandas.DataFrame: The preprocessed Drebin dataset containing only 'B' class.
        dict: A dictionary of metrics collected during preprocessing.
    """
    # Total entries before preprocessing
    initial_entries = input_data.shape[0]

    # Count instances of each class before filtering 'B'
    class_dist_before_filter = input_data["class"].value_counts()

    # Filter only class 'B' instances
    filtered_data = input_data[input_data['class'] == 'B']
    #filtered_data = input_data[input_data['class'] == 'S'] # for class 'S'

    # Metrics collection
    metrics = {
        'initial_entries': initial_entries,
        'class_distribution_before': class_dist_before_filter,
    }

    return filtered_data, metrics

def plot_metrics(metrics):
    """
    Plots the class distribution before filtering.

    Args:
        metrics (dict): A dictionary of preprocessing metrics.
    """
    # Bar chart for class distribution before filtering
    plt.figure(figsize=(10, 5))
    sns.barplot(x=metrics['class_distribution_before'].index, y=metrics['class_distribution_before'].values)
    plt.title("[Drebin] Class Distribution Before Filtering")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

if __name__ == '__main__':
    input_data = pd.read_csv('drebin.csv')  # Load data
    clean_data, metrics = preprocess_data(input_data)  # Preprocess data (filter for 'B')
    clean_data.to_csv('cleaned_drebin_class_B.csv', index=False)  # Save preprocessed data
    # clean_data.to_csv('cleaned_drebin_class_S.csv', index=False)  # For class "S"
    plot_metrics(metrics)  # Plot metrics (class distribution before filtering)