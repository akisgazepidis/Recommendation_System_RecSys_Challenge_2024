import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tqdm.keras import TqdmCallback
import datetime
import os
# Suppress TensorFlow warnings if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages


# Read interaction matrix pickle file
size = 'demo'
fillna_value = '0'
interaction_matrix_file_path = f'./files/pickle/interaction_matrix_{size}_{fillna_value}.pkl'
interaction_matrix_df = pd.read_pickle(interaction_matrix_file_path)
print('Interaction matrix df shape:                      ',interaction_matrix_df.shape)

user_matrix_df_file_path = f'./files/pickle/user_matrix_{size}_{fillna_value}.pkl'
article_matrix_df_file_path = f'./files/pickle/article_matrix_{size}_{fillna_value}.pkl'

user_matrix_df = pd.read_pickle(user_matrix_df_file_path)
article_matrix_df = pd.read_pickle(article_matrix_df_file_path)
print('User embedding df shape:                         ',user_matrix_df.shape)
print('Article embedding df shape:                      ',article_matrix_df.shape)

# Convert the dataframes to numpy arrays
user_vectors = user_matrix_df.values
article_vectors = article_matrix_df.values
interaction_matrix = interaction_matrix_df.values

# Get the indices of the non-zero entries in the interaction matrix
user_idx, article_idx = np.where(interaction_matrix != 0)
read_times = interaction_matrix[user_idx, article_idx]

# Create the input features by concatenating user and article vectors
X = np.hstack((user_vectors[user_idx], article_vectors[article_idx]))
y = read_times

# Use only the first 100 rows for testing
X = X[:100]
y = y[:100]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Define the model
model = Sequential([
    tf.keras.Input(shape=(600,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Prepare TensorBoard callback
log_dir = "files/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
num_epochs = 2
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=16,
                    validation_data=(X_test, y_test),
                    callbacks=[tensorboard_callback, TqdmCallback(verbose=1)],
                    verbose=2)
