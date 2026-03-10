"""
My Model

This file automatically discovers and loads your .h5 model file and .csv data file
from the miner_model directory. Just place your files there and you're ready to go!
The predict function is already provided - it uses the standard 1-hour-ahead pattern.
"""

import os
import glob
import pandas as pd
from tensorflow.keras.models import load_model

# Import helper functions (they handle the standard prediction pattern)
from .helpers import predict_1hour_ahead, prepare_dataframe

# ============================================
# SECTION 1: Load Your Model
# ============================================
# By default, this searches for .h5 files in the miner_model directory
# (excluding LSTM_outside_example/lstm_model.h5).
# You can change the model_path if your model is in a different location.

# Get the miner_model directory (parent of student_models)
current_dir = os.path.dirname(os.path.abspath(__file__))
miner_model_dir = os.path.dirname(current_dir)

# Search for .h5 files in miner_model directory, excluding LSTM_outside_example
model_files = []
for root, dirs, files in os.walk(miner_model_dir):
    # Skip LSTM_outside_example directory
    if 'LSTM_outside_example' in root:
        continue
    for file in files:
        if file.endswith('.h5'):
            model_files.append(os.path.join(root, file))

if not model_files:
    raise FileNotFoundError(
        f"No .h5 model files found in {miner_model_dir}.\n"
        "Please save your model as a .h5 file in the miner_model directory, "
        "or update model_path below to point to your model file."
    )

# Use the first .h5 file found (you can change this to select a specific file)
model_path = model_files[0]

# Load the model
# You can customize this if needed (e.g., change compile=False, add custom_objects, etc.)
model = load_model(model_path, compile=False)


# ============================================
# SECTION 2: Load Your Data  
# ============================================
# By default, this searches for .csv files in the miner_model directory
# (excluding LSTM_outside_example).
# You can change the data_path if your data is in a different location.

# Search for .csv files in miner_model directory, excluding LSTM_outside_example
csv_files = []
for root, dirs, files in os.walk(miner_model_dir):
    # Skip LSTM_outside_example directory
    if 'LSTM_outside_example' in root:
        continue
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

if not csv_files:
    raise FileNotFoundError(
        f"No .csv data files found in {miner_model_dir}.\n"
        "Please save your data as a .csv file in the miner_model directory, "
        "or update data_path below to point to your data file."
    )

# Use the first .csv file found (you can change this to select a specific file)
data_path = csv_files[0]

# Load and prepare data using helper function
df = pd.read_csv(data_path)
data = prepare_dataframe(df)  # Helper handles all the formatting!


# ============================================
# SECTION 3: Predict Function (ALREADY DONE!)
# ============================================
# The predict function uses the standard helper function.
# You can customize the parameters if needed.

def predict(timestamp):
    """
    Predict New England LoadMw 1 hour ahead.
    
    Uses the standard prediction pattern via helper function.
    You can customize the parameters below if needed.
    """
    return predict_1hour_ahead(
        model=model,
        data=data,
        timestamp=timestamp,
        n_steps=12,  # 12 timesteps = 1 hour (for 5-minute data)
        interval_method='fixed',  # Change to 'std' if you have standard error
        interval_std=None  # Set your std_error here if using 'std' method
    )
