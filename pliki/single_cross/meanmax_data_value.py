import pandas as pd

df = pd.read_excel('pretty_single_cross.xlsx', sheet_name='single_cross_hipertuning_1')
filtered_df = df[df['num'] == '_3']
grouped = filtered_df.groupby('data_value')['result'].agg(['mean', 'max'])
sorted_grouped = grouped.sort_values(by='max', ascending=False)
print(sorted_grouped)


