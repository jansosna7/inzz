import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
df = pd.read_excel('analyze.xlsx', header=0)

print(df.keys())


# Filter the data to only include rows where time is between 6000 and 7000 seconds
filtered_df = df[(df['time'] >= 6000) & (df['time'] <= 7000)]

# Group by 'id' and take the latest entry for each group
latest_df = filtered_df.sort_values(by='time').groupby('id').tail(1)

# Save the latest results to an Excel file
latest_df.to_excel('sample.xlsx', index=False)

# Function to plot a graph
def plot_and_save_graph(x_col, y_col, title, filename):
    plt.figure()
    plt.scatter(latest_df[x_col], latest_df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Plot graphs
plot_and_save_graph('epsilon', 'result', 'Epsilon vs Result','Epsilon_vs_Result.png')
plot_and_save_graph('harsh_time', 'result', 'Harsh Time vs Result','Harsh-Time_vs_Result.png')
plot_and_save_graph('harsh_result', 'result', 'Harsh Result vs Result', 'Harsh-Result_vs_Result.png')
