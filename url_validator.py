# url_validator.py
from predict_url import predict_url

def validate_url(url: str):
    prediction, score = predict_url(url)

    category = "SAFE" if prediction == "benign" else "DANGEROUS"
    reasons = [f"Predicted as {prediction.upper()} by CNN-LSTM model"]

    return {
        "url": url,
        "score": round(score, 2),
        "category": category,
        "reasons": reasons
    }
