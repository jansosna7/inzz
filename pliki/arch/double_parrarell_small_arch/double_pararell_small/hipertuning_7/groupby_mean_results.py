import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# Load the Excel file
file_path = 'hipertuning_7.xlsx'
df = pd.read_excel(file_path)
column_names = df.columns.tolist()

# Initialize an empty DataFrame to store the results
results = pd.DataFrame(columns=['Column', 'Unique Value', 'Average'])

# Iterate over columns 1 to 30
for col in range(1,len(column_names)):
    # Group by unique values in the current column and calculate the average of column 0
    averages = df.groupby(df.columns[col])[df.columns[0]].mean().reset_index()
    averages.columns = ['Unique Value', 'Average']
    averages['Column'] = df.columns[col]
    results = pd.concat([results, averages], ignore_index=True)


results.to_excel('averages_results_7.xlsx', index=False)
