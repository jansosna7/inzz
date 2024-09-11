import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel
data = pd.read_excel('results.xlsx', header=0)  # Assuming headers are in the first row


column_names = data.columns.tolist()

for i in range(1,len(column_names)):

    # Assuming the 2nd and 4th columns are variable and result respectively
    variable_column_name = column_names[i]  # Assuming variable is in the 2nd column
    result_column_name = column_names[0]    # Assuming result is in the 4th column

    data = data.sort_values(by=variable_column_name)


    # Extracting variable and result columns
    variable_column = data[variable_column_name]
    result_column = data[result_column_name]

    # Plotting
    plt.scatter(variable_column, result_column)
    plt.xlabel(variable_column_name)
    plt.ylabel(result_column_name)
    plt.title(f'Plot of f({variable_column_name}) = {result_column_name}')
    plt.grid(True)
    plt.savefig("plot_"+str(variable_column_name)+'.png')
    plt.close()
