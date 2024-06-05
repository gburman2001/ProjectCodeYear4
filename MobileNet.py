#Original MobileNet Structure found here, model is adapted for this project - https://towardsdatascience.com/building-mobilenet-from-scratch-using-tensorflow-ad009c5dd42c
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras as ks
from keras._tf_keras.keras.layers import Dense
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_absolute_percentage_error, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import root_mean_squared_error
from math import sqrt

# Read the data from the csv file
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare the data by splitting between features and labels
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

# Define the model shape
modelMobile = ks.Sequential([
    Dense(64, activation='relu', input_shape=(3,)),  # Input shape matches the number of features (R,G,B values)
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # Output shape matches the number of classes (non-ship or ship)
])

# Compile the model
modelMobile.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
modelMobile.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))


# Predict the test set
y_pred_probs = modelMobile.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

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

# Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creates the results dictionary
results = {
        'Model': "MobileNet", # Sets the model name
        # Sets the result metrics of the models
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
