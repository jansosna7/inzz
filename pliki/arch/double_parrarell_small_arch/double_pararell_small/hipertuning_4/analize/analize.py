import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel
data = pd.read_excel('hipertuning_4.xlsx', header=0)  # Assuming headers are in the first row

column_names = data.columns.tolist()

# Loop over each pair of columns
for i in range(1, len(column_names)):
    for j in range(i, len(column_names)):
        if i != j:
            col_a = column_names[i]
            col_b = column_names[j]
            # Create new columns for a*b and a/b
            data[f'{col_a}_times_{col_b}'] = data[col_a] * data[col_b]
            data[f'{col_a}_div_{col_b}'] = data[col_a] / data[col_b]

# Save the DataFrame to an Excel file (optional)
data.to_excel('hipertuning_4_analize.xlsx', index=False)
