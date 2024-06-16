import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tqdm.keras import TqdmCallback
import os
# Suppress TensorFlow warnings if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import warnings
warnings.filterwarnings("ignore", message="NotOpenSSLWarning")


# Dummy data for testing with values between -1 and 1
X_train = np.random.uniform(low=-1, high=1, size=(100, 600))
y_train = np.random.uniform(low=-1, high=1, size=(100,))
X_test = np.random.uniform(low=-1, high=1, size=(20, 600))
y_test = np.random.uniform(low=-1, high=1, size=(20,))

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
