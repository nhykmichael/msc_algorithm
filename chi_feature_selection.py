
import pandas as pd
from scipy.stats import chi2_contingency
import json
import matplotlib.pyplot as plt

def chi_square_feature_selection(input_data, target_variable, significance_level=0.05):
    """
    Performs Chi-square feature selection.

    Args:
        input_data (pandas.DataFrame): The input dataset.
        target_variable (str): The name of the target variable.
        significance_level (float, optional): The significance level for feature selection. Defaults to 0.05.

    Returns:
        dict: A dictionary of selected features with chi-square values and p-values.
    """  
    features = input_data.columns.drop(target_variable)
    selected_features = {}
    for feature in features:
        contingency_table = pd.crosstab(input_data[feature], input_data[target_variable])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value <= significance_level:
            selected_features[feature] = {'chi2_stat': chi2_stat, 'p_value': p_value}
    return selected_features

if __name__ == '__main__':
    input_data = pd.read_csv("drebin.csv")
    target_variable = "class"
    selected_features = chi_square_feature_selection(input_data, target_variable)
    
    # Save features to JSON
    with open('chi_features_by_value.json', 'w') as json_file:
        json.dump(selected_features, json_file, indent=4)
    
    # Sort features by chi-square value and plot top 20
    sorted_features = sorted(selected_features.items(), key=lambda item: item[1]['chi2_stat'], reverse=True)
    top_features = sorted_features[:20]
    
    features = [item[0] for item in top_features]
    chi2_values = [item[1]['chi2_stat'] for item in top_features]
    
    plt.figure(figsize=(8, 6))
    plt.barh(features, chi2_values, color='skyblue')
    plt.xlabel('Chi-Square Value')
    plt.ylabel('Features')
    plt.title('Top 20 Features by Chi-Square Value')
    plt.gca().invert_yaxis()  # Invert y-axis to display the highest value on top
    plt.show()
    
    print("Selected features saved to 'chi_features_by_value.json' and top 20 features plotted.")
