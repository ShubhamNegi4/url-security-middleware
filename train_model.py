# CNN + LSTM Model to Classify Malicious URLs

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle

# 📥 Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("malicious_phish.csv")

# Use correct column names
if "url" in df.columns and "type" in df.columns:
    df = df[["url", "type"]].dropna()
    df.columns = ["url", "label"]  # only if needed

else:
    raise Exception("❌ Dataset does not contain expected columns ['URL', 'type'].")

df = df.dropna()
df["label"] = df["label"].str.lower()

# 🔐 Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# 🔧 Hyperparameters
MAX_LEN = 200
VOCAB_SIZE = 10000
EMBED_DIM = 128

# 🔡 Tokenize URLs
tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, lower=True)
tokenizer.fit_on_texts(df["url"])
sequences = tokenizer.texts_to_sequences(df["url"])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = df["label_encoded"].values

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 🧠 Build CNN + LSTM model
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 🛑 Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 🚀 Train model
print("🚀 Training model...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,                # ✅ Reduced from 4070
    batch_size=128,
    callbacks=[early_stop],   # ✅ Stops early if no improvement
    verbose=1
)

# 💾 Save model and tokenizer
os.makedirs("saved_models", exist_ok=True)
print("💾 Saving model and tokenizer...")
model.save("saved_models/url_cnn_lstm_model.h5")
with open("saved_models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Training complete. Model saved to 'saved_models/url_cnn_lstm_model.h5'")
