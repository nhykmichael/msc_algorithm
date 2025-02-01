import pandas as pd

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Load the first CSV file
df1 = pd.read_csv('real_legitimate_v1.csv')
dfe1 = df1.drop(columns=['Package', 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
dfe1 = dfe1.iloc[:,:-11]

# Load the second CSV file
df2 = pd.read_csv('real_malware_v1.csv')
dfe2 = df2.drop(columns=['Package', 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
dfe2 = dfe2.iloc[:,:-11]

# Merge the two DataFrames
merged_df = pd.concat([df1, df2])

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('all.csv', index=False)
