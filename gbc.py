from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np


# Load dataset
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare data from the csv data
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the Gradient Boosting Classifier
classifierGBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
classifierGBC.fit(X_train, y_train)

# Make predictions on the test set of data
y_pred = classifierGBC.predict(X_test)

# Evaluate the model
# Creates the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Creates the precision score
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Creates the recall score
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Creates the f1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Creates the MAE score
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Creates the MAPE score
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)

# Creates the RMSE score
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Creates a confusion matrix to verify results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creates a results dictionary
results = {
        # Sets model name
        'Model': "GBC",
        # Sets the models metric scores
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