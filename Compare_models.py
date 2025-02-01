import matplotlib.pyplot as plt
import numpy as np

'''
Author: MN Ahimbisibwe
SN: 217005435
Varsity: University of Johannesburg
Course: MSC CS
'''

# Model data
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
metrics_50_epochs = {
    'Precision': [0.9546, 0.9956, 0.9009, 0.9230, 0.9358],
    'Recall': [0.8192, 0.9476, 0.9438, 0.9288, 0.9732],
    'F-1': [0.8817, 0.9709, 0.9219, 0.9259, 0.9541],
    'Accuracy': [0.8347, 0.9573, 0.8797, 0.9213, 0.9505]
}
metrics_100_epochs = {
    'Precision': [0.9631, 0.9926, 0.9184, 0.8651, 0.9523],
    'Recall': [0.9030, 0.9616, 0.9840, 0.8631, 0.9804],
    'F-1': [0.9321, 0.9768, 0.9501, 0.8631, 0.9661],
    'Accuracy': [0.9011, 0.9657, 0.9223, 0.8562, 0.9636]
}

# Plotting Precision
plt.figure(figsize=(10, 5))
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, metrics_50_epochs['Precision'], width, label='50 Epochs')
plt.bar(x + width/2, metrics_100_epochs['Precision'], width, label='100 Epochs')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Model Comparison by Precision')
plt.xticks(x, models)
plt.legend()
plt.savefig("model_comparison_precision.png")
plt.show()

# Plotting Recall
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, metrics_50_epochs['Recall'], width, label='50 Epochs')
plt.bar(x + width/2, metrics_100_epochs['Recall'], width, label='100 Epochs')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Model Comparison by Recall')
plt.xticks(x, models)
plt.legend()
plt.savefig("model_comparison_recall.png")
plt.show()

# Plotting F-1 Score
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, metrics_50_epochs['F-1'], width, label='50 Epochs')
plt.bar(x + width/2, metrics_100_epochs['F-1'], width, label='100 Epochs')
plt.xlabel('Models')
plt.ylabel('F-1 Score')
plt.title('Model Comparison by F-1 Score')
plt.xticks(x, models)
plt.legend()
plt.savefig("model_comparison_f1.png")
plt.show()

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, metrics_50_epochs['Accuracy'], width, label='50 Epochs')
plt.bar(x + width/2, metrics_100_epochs['Accuracy'], width, label='100 Epochs')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison by Accuracy')
plt.xticks(x, models)
plt.legend()
plt.savefig("model_comparison_accuracy.png")
plt.show()
