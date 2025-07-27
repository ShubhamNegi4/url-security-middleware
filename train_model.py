import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 📥 Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("malicious_phish.csv")

# ✅ Use correct column names
if "url" in df.columns and "type" in df.columns:
    df = df[["url", "type"]].dropna()
    df.columns = ["url", "label"]
else:
    raise Exception("❌ Dataset does not contain expected columns ['url', 'type'].")

df["label"] = df["label"].str.lower()

print("\n📊 Class distribution:")
print(df["label"].value_counts())

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

# 🧠 Build model (CNN + LSTM)
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  # Softmax for multi-class
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 🛑 Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ⚖️ Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 🚀 Train model
print("\n🚀 Training model...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# 📈 Evaluate model
print("\n✅ Evaluating model on test set...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# 📋 Classification report
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 💾 Save everything
os.makedirs("saved_models", exist_ok=True)
print("\n💾 Saving model and preprocessing tools...")
model.save("saved_models/url_cnn_lstm_model.keras")
with open("saved_models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✅ Training complete. Model and tools saved in 'saved_models/'")
