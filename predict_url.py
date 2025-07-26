# predict_url.py - Predict malicious or benign URL using trained model

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("saved_models/url_cnn_lstm_model.h5")
with open("saved_models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

def predict_url(url: str) -> str:
    sequence = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prob = model.predict(padded)[0][0]
    label = "malicious" if prob > 0.5 else "benign"
    return f"Prediction: {label.upper()} (score={prob:.2f})"

# CLI or test block
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_url = sys.argv[1]
        print(f"🔍 {input_url}\n{predict_url(input_url)}")
    else:
        # test mode
        test_urls = [
            "http://example.com/login",
            "http://free-bitcoin.ru/get-rich-now",
            "https://secure-login.ph1sh.xyz/index.php?id=123",
            "https://www.google.com"
        ]
        for url in test_urls:
            print(f"🔍 {url}\n{predict_url(url)}\n")
