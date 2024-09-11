import pandas as pd

df = pd.read_excel('pretty_single_cross.xlsx', sheet_name='single_cross_hipertuning_1')
filtered_df = df[df['num'] == '_3']
grouped = filtered_df.groupby('epsilon')['result'].agg(['mean', 'max'])

print(grouped.loc[[0.1, 0.01]])

filtered_df = df[df['num'] == '_4']
grouped = filtered_df.groupby('epsilon')['result'].agg(['mean', 'max'])

print(grouped.loc[[0.1, 0.01]])
