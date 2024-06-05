from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare data for training and testing
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Define the individual classifiers for the Ensemble
gbc = GradientBoostingClassifier()
rf = RandomForestClassifier()

# Create a voting classifier for the combined model, by using hard we are more likely to favour the stronger performing model
voting_clf = VotingClassifier(estimators=[
    ('gbc', gbc),
    ('rf', rf)], 
    voting='hard')

# Train the voting classifier on the training data
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Calculates the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculates the precision score
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculates the recall score
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculates the f1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculates the MAE score
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Calculates the MAPE score
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)

# Calculates the RMSE score
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Creates the confusin matrix to verify results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creates the model dictionary
results = {
        'Model': "Ensemble", # Sets the model name
        # Sets all the metric results of the model
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse
    }
    
# Convert results to DataFrame
results_df = pd.DataFrame([results])
csv_file = 'model_results.csv'
# Append results to csv file
results_df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)