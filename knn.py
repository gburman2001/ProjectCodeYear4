# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np


# Load dataset from the csv file
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare data from the csv file
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors manually

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test)

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

# Creates the confusion matrix to verify results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creates the results dictionary
results = {
        'Model': "KNN", # Sets the model name
        # Sets the model metrics
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
# Append results to CSV file
results_df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)