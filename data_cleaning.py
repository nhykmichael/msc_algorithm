import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Metrices
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

def preprocess_data(input_data):
    """
    Preprocesses the Drebin and KronoDroid datasets by removing duplicates, 
    filtering for benign instances, handling missing values, and normalizing the data.

    Args:
        input_data (pandas.DataFrame): The input Drebin dataset.

    Returns:
        pandas.DataFrame: The preprocessed Drebin dataset.
    """
    # Total entries before preprocessing
    initial_entries = input_data.shape[0];
    # Remove duplicates
    
      
    input_data = input_data.drop_duplicates()
    duplicates_removed = initial_entries - input_data.shape[0]
    print("Duplicates removed =", duplicates_removed)
    
    # Count instances of each class before filter 'B' class
    class_dist_before_filter = input_data["class"].value_counts()
    
    # Count missing values before imputation
    missing_values_before = input_data.isnull().sum()
    
    # Handle missing values using most frequent imputation
    data_imputation = input_data.fillna(input_data.mode().iloc[0])
        
    # Missing values after imputation
    missing_values_after = data_imputation.isnull().sum()
    
    
    # Normalize the data (scaling) if needed i.e if data have different numerical categories
    # For example age in years and income in dollars
    '''
    numerical_features = filtered_data.select_dtypes(include=['number']).columns
    filtered_data[numerical_features] = StandardScaler().fit_transform(filtered_data[numerical_features])
    ****************** OR *******************************************************************************
    numerical_features = data_imputation.select_dtypes(include=['number']).columns
    normalized_data = data_imputation.copy()
    normalized_data[numerical_features] = (data_imputation[numerical_features] - 
                                           data_imputation[numerical_features].min()) / \
                                          (data_imputation[numerical_features].max() - 
                                           data_imputation[numerical_features].min())
    return normalized_data
    '''
    
    # Metrics collection
    metrics = {
        'initial_entries': initial_entries,
        'duplicates_removed': duplicates_removed,
        'class_distribution_before': class_dist_before_filter,
        'missing_values_before': missing_values_before,
        'missing_values_after': missing_values_after
    }
    
    
    return data_imputation, metrics
    
def class_labler(data):
    """
    Encodes the class labels in the dataset.
    
    Args:
        data (pandas.DataFrame): The input Drebin/KronoDroid dataset after preprocessing.
    
    Returns:
        pandas.DataFrame: The dataset with encoded class labels.
    """
    # Encord class label assume is categorical not numerical
    label_encoder = LabelEncoder()
    data['class'] = label_encoder.fit_transform(data['class'])
    return data
    
def plot_metrics(metrics, dataset):
    """
    Plots the metrics to provide a visual overview of the preprocessing steps.
    
    Args:
        metrics (dict): A dictionary of preprocessing metrics.
    """
    # Bar chart for class distribution before filtering
    plt.figure(figsize=(10, 5))
    sns.barplot(x=metrics['class_distribution_before'].index, y=metrics['class_distribution_before'].values)
    plt.title("Class Distribution Before Filtering")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()
    
    # Pie chart for duplicates
    plt.figure(figsize=(5, 5))
    plt.pie([metrics['initial_entries'] - metrics['duplicates_removed'], metrics['duplicates_removed']],
            labels=['Non-duplicates', 'Duplicates'], autopct='%1.1f%%', startangle=90)
    plt.title(dataset + " Duplicate Entries")
    plt.show()
    
    # Bar chart for missing values before and after imputation
    missing_df = pd.DataFrame({
        'Missing Before': metrics['missing_values_before'],
        'Missing After': metrics['missing_values_after']
    })
    
    missing_df.plot(kind='bar', figsize=(10, 5))
    plt.title("Missing Values Before and After Imputation")
    plt.xlabel("Features")
    plt.ylabel("Count")
    plt.show()
    

if __name__ == '__main__':
    input_data = pd.read_csv('Drebin.csv')  # Load data
    clean_data, metrics = preprocess_data(input_data)  # Preprocess data
    #clean_data.to_csv('cleaned_drebin.csv', index=False)  # Save preprocessed data
    #cleaned_data = pd.read_csv('cleaned_drebin.csv')
    #labeled_data = class_labler(cleaned_data)
    #labeled_data.to_csv('cleaned_drebin_CL.csv', index=False);
    
    plot_metrics(metrics, "Drebin")
    encoded_data = class_labler(clean_data)  # Encode class labels
