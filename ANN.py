from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare data from the csv file
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creates the Artificial Neural Network 
classifierANN = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam')

# Trains the classifier with the testing data and labels
classifierANN.fit(X_train, y_train)
# Makes label predictions from the testing data
y_pred = classifierANN.predict(X_test)

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

# Prints out a matrix of the results, allows for personal verification of the results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creates a results dictionary to add to the results csv
results = {
        'Model': "ANN", # Sets the model name
        # Sets the different metrics of this model
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