# predict_url.py

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse

# 🔃 Load model and pre-processing tools
model = tf.keras.models.load_model("saved_models/url_cnn_lstm_model.keras")
with open("saved_models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("saved_models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 200  # Must match training
THRESHOLD = 0.80  # Confidence threshold

# ✅ Trusted allowlist of known safe domains
allowlist = [
    "www.google.com",
    "www.microsoft.com",
    "github.com",
    "www.wikipedia.org"
]

# 🔍 Input URLs for prediction
urls = [
    "http://example.com/login",
    "http://free-bitcoin.ru/get-rich-now",
    "https://secure-login.ph1sh.xyz/index.php?id=123",
    "https://www.google.com"
]

print()
for url in urls:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    print(f"🔍 {url}")
    
    # ✅ Check against allowlist
    if domain in allowlist:
        print("✅ SAFE (trusted domain - allowlisted)\n")
        continue

    # 🔮 Predict using model
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)

    class_index = np.argmax(prediction, axis=1)[0]
    class_label = label_encoder.inverse_transform([class_index])[0]
    confidence = prediction[0][class_index]

    # ✅ Final decision based on label and confidence
    if class_label == "benign" and confidence >= THRESHOLD:
        print(f"✅ SAFE (benign with confidence: {confidence:.2f})\n")
    else:
        print(f"⚠️ MALICIOUS (detected as: {class_label.upper()}, confidence: {confidence:.2f})\n")
