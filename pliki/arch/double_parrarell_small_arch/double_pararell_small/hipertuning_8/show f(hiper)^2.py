import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from Excel file
data = pd.read_excel("hipertuning_8.xlsx")

# Separate the score from parameters
score = data.iloc[:, 0]  # Assuming the score is in the first column
parameters = data.iloc[:, 1:8]  # Assuming parameters start from the second column

# Create a scatter plot matrix using Seaborn
scatter_matrix = sns.pairplot(parameters, diag_kind='kde')  # Use 'kde' for kernel density estimates on diagonal

# Add score as color
for i, j in zip(*plt.np.triu_indices_from(scatter_matrix.axes, 1)):
    scatter_matrix.axes[i, j].scatter(parameters.iloc[:, j], parameters.iloc[:, i], c=score, cmap='viridis', alpha=0.5)

# Show plot
    plt.savefig("plot2_"+"10"+'.png')
    plt.close()
