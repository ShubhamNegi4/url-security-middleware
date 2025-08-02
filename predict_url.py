# predict_url.py

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse

# 🔃 Load model and pre-processing tools
model = tf.keras.models.load_model(r"c:/Users/bhatt/OneDrive/Documents/gggg/url-security-middleware/url-security-middleware/saved_models/url_cnn_lstm_model.keras")
with open(r"c:/Users/bhatt/OneDrive/Documents/gggg/url-security-middleware/url-security-middleware/saved_models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open(r"c:/Users/bhatt/OneDrive/Documents/gggg/url-security-middleware/url-security-middleware/saved_models/label_encoder.pkl", "rb") as f:
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
    # Valid benign
    "https://www.google.com",
    "https://github.com",
    "https://university.edu/home?ref=42",
    # Malicious
    "http://free-bitcoin.ru/get-rich-now",
    "https://secure-login.ph1sh.xyz/index.php?id=123",
    "http://malware-download.biz/<script>alert(1)</script>",
    # Invalid/non-URL
    "hmy name name",
    "just some random text",
    "1234567890",
    # Edge-case
    "http://192.168.1.1",
    "ftp://example.com/resource",
    "http://clickjack.tk/?q=' OR 1=1 --",
    "http://example.com/%3Csvg/onload=alert(1)%3E"
]

def predict_url(url: str):
    """Predict class for a URL or string. Returns top-2 classes and confidences, with explanations for not_a_url/edge_case."""
    # Convert HttpUrl object to string if needed
    if hasattr(url, '__str__'):
        url = str(url)

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    scheme = parsed.scheme.lower()

    # ✅ Check against allowlist (only for valid URLs with netloc)
    if domain in allowlist and domain != "":
        return {
            "prediction": "benign",
            "confidence": 1.0,
            "secondary": None,
            "explanation": "Trusted domain (allowlisted)."
        }

    # ⚠️ Input validation: check if input is a valid URL
    is_valid_url = bool(domain) and bool(scheme)
    warning = None
    if not is_valid_url:
        warning = "⚠️ Input is not a valid URL. Prediction may not be meaningful."

    # 🔮 Predict using model regardless
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)[0]

    top2_idx = prediction.argsort()[-2:][::-1]
    top1, top2 = top2_idx[0], top2_idx[1]
    class1 = label_encoder.inverse_transform([top1])[0]
    class2 = label_encoder.inverse_transform([top2])[0]
    conf1 = float(prediction[top1])
    conf2 = float(prediction[top2])

    explanation = None
    if class1 == "not_a_url":
        explanation = "Input does not appear to be a URL."
    elif class1 == "edge_case":
        explanation = "Input is a rare/edge-case URL (e.g., IP, FTP, encoded, or partial)."
    elif warning:
        explanation = warning
    else:
        explanation = None

    return {
        "prediction": class1,
        "confidence": conf1,
        "secondary": {"class": class2, "confidence": conf2},
        "explanation": explanation
    }

# Demo code for testing
if __name__ == "__main__":
    print()
    for url in urls:
        print(f"🔍 {url}")
        result = predict_url(url)
        if isinstance(result, dict):
            print(f"  Prediction      : {result['prediction'].upper()}")
            print(f"  Confidence      : {result['confidence']:.2f}")
            if result['secondary']:
                print(f"  Runner-up Class : {result['secondary']['class'].upper()} (confidence: {result['secondary']['confidence']:.2f})")
            if result['explanation']:
                print(f"  Explanation     : {result['explanation']}")
            print()
        else:
            # fallback for any tuple output
            print(f"  Prediction      : {result[0]}")
            print(f"  Confidence      : {result[1]:.2f}\n")
