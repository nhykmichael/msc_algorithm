import pandas as pd

# Load the Drebin dataset
drebin_file = 'drebin.csv'  # Replace with your file path if different
df_drebin = pd.read_csv(drebin_file)

# Check if the 'class' column exists
if 'class' not in df_drebin.columns:
    raise ValueError("The dataset does not contain a 'class' column.")

# Separate the dataset into class B and class S
df_class_b = df_drebin[df_drebin['class'] == 'S']
df_class_s = df_drebin[df_drebin['class'] == 'S']

# Ensure at least 600 samples, with a mix of B and S
n_samples_b = min(1000, len(df_class_b))  # Take up to 300 samples from class B
n_samples_s = min(1000, len(df_class_s))  # Take up to 300 samples from class S

# Sample from each class
df_class_b_sample = df_class_b.sample(n=n_samples_b, random_state=42)
df_class_s_sample = df_class_s.sample(n=n_samples_s, random_state=42)

# Combine the samples into a single dataset
df_mixed = pd.concat([df_class_b_sample, df_class_s_sample]).sample(frac=1, random_state=42)  # Shuffle the dataset

# Save the mixed dataset to a CSV file
output_file = 'drebin_s_2000.csv'
df_mixed.to_csv(output_file, index=False)

print(f"Mixed dataset with 6650 samples saved to {output_file}")
