import pandas as pd

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Load the dataset
dataset = pd.read_csv('real_legitimate_v1.csv')

# Statistical summary of the dataset
summary = dataset.describe()

# Display summary
print(summary)

import matplotlib.pyplot as plt
import seaborn as sns



# Example: Visualizing the distribution of system calls
plt.figure(figsize=(10,6))
sns.histplot(dataset['execve'], bins=30, kde=True, color='blue', label='execve')
plt.title('Distribution of execve System Call')
plt.xlabel('execve Calls')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Example: Replace 'malware_family' with the actual column name from your dataset
plt.figure(figsize=(12,6))
malware_family_counts = dataset['MalFamily'].value_counts()
malware_family_counts.plot(kind='bar', color='orange')
plt.title('Malware Family Distribution')
plt.xlabel('Malware Family')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.show()

# Replace 'normal', 'dangerous', 'signature' with permission columns
permissions_distribution = dataset[['normal', 'dangerous', 'signature']].sum()

plt.figure(figsize=(8,8))
plt.pie(permissions_distribution, labels=permissions_distribution.index, autopct='%1.1f%%', colors=['lightblue', 'red', 'green'])
plt.title('Distribution of Permissions Categories')
plt.show()

'''
# Select relevant system calls (e.g., top 5 based on importance)
important_calls = ['execve', 'getuid32', 'getgid32', 'readahead', 'open']

# Compute the correlation matrix
corr_matrix = dataset[important_calls].corr()

# Plot the heatmap with annotations and highlighting strong correlations (replace threshold with your desired value)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='Reds', linewidths=0.5, vmax=0.8)  # Highlight values above 0.8

# Add formatting for strong correlations (example using matplotlib)
for i, row in corr_matrix.iterrows():
  for j, value in row.items():
    if abs(value) > 0.8:
      plt.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Correlation Matrix of Important System Calls')
plt.show()
'''
# Select relevant system call and permission columns for correlation analysis
system_calls = ['execve', 'getuid32', 'getgid32', 'readahead']  # Example columns

# Compute the correlation matrix
corr_matrix = dataset[system_calls].corr()

# Plot the heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of System Calls')
plt.show()

# Replace 'Malware' with the column indicating malware/benign status
plt.figure(figsize=(10,6))
sns.boxplot(x='Malware', y='execve', data=dataset)
plt.title('System Call (execve) Variability Between Malware and Benign Samples')
plt.xlabel('Malware Status')
plt.ylabel('execve System Call Frequency')
plt.show()

# Replace 'Malware' with the actual column name that indicates the malware/benign class
class_distribution = dataset['Malware'].value_counts()

plt.figure(figsize=(8,8))
plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', colors=['green', 'red'])
plt.title('Malware vs Benign Class Distribution')
plt.show()


