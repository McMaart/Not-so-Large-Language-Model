import matplotlib.pyplot as plt

# Data for the first table
categories = ["Grammar", "Spelling", "Consistency", "Story/Plot", "Creativity", "Style"]
eval_loss_1 = [1.89, 1.67, 1.47, 1.43, 1.39]
scores_1 = {
    "Grammar": [7.61, 8.32, 8.41, 8.47, 9.1],
    "Spelling": [9.68, 9.69, 9.73, 9.72, 9.73],
    "Consistency": [6.03, 6.52, 6.33, 6.54, 7.97],
    "Story/Plot": [4.76, 4.78, 4.78, 4.79, 5.61],
    "Creativity": [4.65, 4.61, 4.60, 4.65, 5.22],
    "Style": [5.41, 5.41, 5.43, 5.51, 6.20]
}

# Data for the second table
eval_loss_2 = [1.65, 1.15]
scores_2 = {
    "Grammar": [7.49, 9.13],
    "Spelling": [9.71, 9.78],
    "Consistency": [5.47, 8.14],
    "Story/Plot": [4.71, 5.65],
    "Creativity": [4.90, 5.24],
    "Style": [5.17, 6.18]
}

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, category in enumerate(categories):
    axes[i].scatter(eval_loss_1, scores_1[category], color='blue', label='Transformer Models')
    axes[i].scatter(eval_loss_2, scores_2[category], color='red', label='GPT-4 Models')
    axes[i].set_title(category)
    axes[i].set_xlabel('Eval Loss')
    axes[i].set_ylabel('Score')
    axes[i].legend()

plt.tight_layout()
plt.show()
