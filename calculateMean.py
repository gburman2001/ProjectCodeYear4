import pandas as pd

# Reads the results csv file
df = pd.read_csv('model_results.csv')

# Calculates the mean by each rows 'Model'
meanDF = df.groupby('Model').mean().reset_index()

# Save resutls to new CSV file
meanDF.to_csv('mean_results.csv', index=False)

# Test to see code is working
print(meanDF)
