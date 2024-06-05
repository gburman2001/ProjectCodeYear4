#Original CNN structure found here and adapted for this project - https://www.tensorflow.org/tutorials/images/cnn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras as ks
from keras._tf_keras.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Input
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_absolute_percentage_error, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import root_mean_squared_error
from math import sqrt

# Reads the csv file
data = pd.read_csv('selected_rows_band4-6d6fdd0c-44ae-4b01-ba92-647049601282 copy.csv')

# Prepare the data from the csv file
X = data[['R', 'G', 'B']].values.astype(np.float32)
y = data['label'].values.astype(np.int64)

# Reshape X to add an extra dimension
X = X.reshape(X.shape[0], X.shape[1], 1)

# Creates the training set of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
modelCNN = ks.Sequential([
    Input(shape=(X_train.shape[1], 1)), # Sets the shape of the data
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=1),
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'), #Sets the output
])

# Compile the model, creating it for the experiment
modelCNN.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model using the training and testing set
modelCNN.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Predict the test set, creating the predicitions used for the testing
y_pred_probs = modelCNN.predict(X_test)
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
rmse = sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE:", rmse)

# Creates a confusion matrix to check the results 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Creates the results dictionary
results = {
        'Model': "CNN Model", # Sets the model name
        # Sets the results of the model metrics to the dictionary
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