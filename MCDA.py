import pandas as pd

# Reads the mean results csv file
df = pd.read_csv('mean_results.csv')

# Sets the weights of each of the metrics with higher emphasis on MAE and RMSE
criteria_weights = {'Accuracy': 0.15, 'Precision': 0.15, 'Recall': 0.15, 'F1 Score': 0.15, 'MAE': 0.2, 'MAPE': 0.0, 'RMSE': 0.2}
# Sets criteria for which metrics are wanted to be lower scores
low_criteria = ['MAE', 'MAPE', 'RMSE'] 

# Normalize the data for the MCDA
for criterion in criteria_weights.keys():
    if criterion in low_criteria:#If the data is considered better if it is closer to 0
        df[criterion + '_norm'] = df[criterion].min() / df[criterion]
    else:
        df[criterion + '_norm'] = df[criterion] / df[criterion].max()

# Calculate weighted scores
df['Score'] = sum(df[criterion + '_norm'] * weight for criterion, weight in criteria_weights.items())

# Creates 'Rank' based on each score which in turn tells which model is best suited for the results
df['Rank'] = df['Score'].rank(ascending=False)


new_columns = ['Model'] + [criterion + '_norm' for criterion in criteria_weights.keys()] + ['Score', 'Rank']# Adds score and rank to the existing data for the user to assess each model with
new_df = df[new_columns]# Adds the rank to the existing dataframe to create a new dataframe

# Creates a new csv file of the results.
new_df.to_csv('MCDA_Results.csv', index=False)
