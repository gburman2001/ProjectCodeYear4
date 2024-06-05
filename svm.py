from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np


# Load dataset
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare data
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creates the SVM classifier
classifierSVM = SVC(kernel='linear') 

# Train the classifier with the training data
classifierSVM.fit(X_train, y_train)

# Predict on the test set of data
y_pred = classifierSVM.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the precision of the model
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate the recall of the model
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate the f1 score of the model
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculate the MAE of the model
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Calculate the MAPE of the model
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)

# Calculate the RMSE of the model
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Print confusion matrix of the results (For my verification) 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creates a dictionary for the results 
results = {
        'Model': "SVM", # Sets model name
        # Sets Model results
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