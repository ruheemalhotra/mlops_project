# ======================================
# 0. Setup (IMPORTANT for CI/CD)
# ======================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# 1. Load Dataset

data_path = "data/blood_cell_anomaly_detection.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print("Original Shape:", df.shape)
# 2. Preprocessing

# Keep only numeric columns
df = df.select_dtypes(include=['float64', 'int64'])

# Fill missing values
df = df.fillna(df.mean())

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

print("Processed Shape:", data_scaled.shape)

# Train-test split
X_train, X_test = train_test_split(
    data_scaled, test_size=0.2, random_state=42
)

# 3. Build Autoencoder Model

input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),  # bottleneck
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

model.summary()


# 4. Train Model

history = model.fit(
    X_train, X_train,
    epochs=10,  # Reduced for CI speed
    batch_size=32,
    validation_data=(X_test, X_test),
    shuffle=True,
    verbose=1
)

# 5. Reconstruction Error

reconstructions = model.predict(data_scaled)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

# 6. Threshold Calculation

threshold = np.mean(mse) + 2 * np.std(mse)
print("Threshold:", threshold)

# Detect anomalies
anomalies = mse > threshold
print("Total anomalies detected:", np.sum(anomalies))

# 7. Save Model + Scaler + Threshold

os.makedirs("model", exist_ok=True)

model.save("model/autoencoder.h5")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(threshold, "model/threshold.pkl")

print(" Model, Scaler, Threshold saved successfully!")


