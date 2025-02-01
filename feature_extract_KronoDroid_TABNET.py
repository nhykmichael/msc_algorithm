import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Load KronoDroid dataset
file_path = "all.csv"  # Replace with your file path
data = pd.read_csv(file_path)
data.drop(columns=['Package', 'sha256', 'EarliestModDate', 'HighestModDate', 'MalFamily'])
data.iloc[:,:-11]

data.fillna(method='ffill', inplace=True)


# Separate features and target
X = data.drop(columns=["Malware"])
y = data["Malware"]

# Label encoding for categorical variables (if any)
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# Initialize and train TabNet model
clf = TabNetClassifier()
clf.fit(
    X_train_np, y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_name=["valid"],
    eval_metric=["accuracy"],
    max_epochs=25,
    patience=10,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Extract feature importance
feature_importances = clf.feature_importances_

# Save feature importance to JSON
feature_importance_dict = {
    "features": list(X.columns),
    "importances": feature_importances.tolist()
}

output_file = "TabNet_feature_importance.json"
with open(output_file, "w") as f:
    json.dump(feature_importance_dict, f, indent=4)

print(f"Feature importances saved to {output_file}")
