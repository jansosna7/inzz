import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
df = pd.read_excel('analyze.xlsx', header=0)

print(df.keys())

# Create a plot
fig, ax = plt.subplots()

# Group the data by 'id' and plot each group
for key, grp in df.groupby(['id']):
    ax.plot(grp['time'], grp['result'], label=f"{key} (e={grp['epsilon'].iloc[0]}, h_time={grp['harsh_time'].iloc[0]}, h_result={grp['harsh_result'].iloc[0]})")

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Result')
ax.set_title('Result over Time for each ID')

# Add a legend
ax.legend()

# Display the plot
plt.show()
