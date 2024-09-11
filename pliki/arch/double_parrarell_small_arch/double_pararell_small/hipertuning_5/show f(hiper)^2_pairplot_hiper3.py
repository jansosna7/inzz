import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from Excel file
data = pd.read_excel("hipertuning_5_2.xlsx")

# Separate the score from parameters
score = data.iloc[:, 0]  # Assuming the score is in the first column
parameters = data.iloc[:, 10:19]  # Assuming parameters start from the second column

# Combine the score and parameters into a single DataFrame
combined_data = pd.concat([parameters], axis=1)

# Create a pairplot using Seaborn
pair_plot = sns.pairplot(combined_data, diag_kind='kde', plot_kws={'alpha': 0.5})

# Add color to each point based on the score
for ax in pair_plot.axes.flat:
    if ax is not None:
        for artist in ax.collections:
            artist.set_alpha(0.5)  # Set transparency
            artist.set_array(score)  # Set the score as the color data

# Manually create a colorbar based on the scatter plot's colormap
fig = plt.gcf()
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
sm.set_array(score)
cbar_ax = fig.add_axes([.92, .3, .02, .4])  # Adjust the position and size of the colorbar
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Score')

# Save the plot as an image without displaying it
plt.savefig("pairplot2_with_colorbar.png", dpi=300)  # You can change the filename and DPI as needed

# Close the plot to release memory
plt.close()
