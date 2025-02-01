import numpy as np
import pandas as pd

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Load your dataset from a CSV file (replace 'your_file.csv' with the actual file name)
data = pd.read_csv('zero_day.csv')

# Ensure the DataFrame has 215 columns
TARGET_COLUMNS = 215
current_columns = data.shape[1]
missing_columns = TARGET_COLUMNS - current_columns

if missing_columns > 0:
    for i in range(missing_columns):
        data[f"Missing_Column_{i+1}"] = 0  # Fill missing columns with zeroes

# Save the updated DataFrame back to a CSV file (optional)
data.to_csv('Zero_day_padded.csv', index=False)

# Verify the new shape
print("New shape:", data.shape)