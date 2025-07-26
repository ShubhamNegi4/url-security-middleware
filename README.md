# 🧠 DeepSecure Proxy

**Real-Time URL Risk Analysis Middleware for Proxy Servers using Machine Learning + Rule-Based Detection**

> A modular, Python-powered URL classification and validation system designed for seamless integration into proxy infrastructures. It inspects, scores, and explains the threat level of URLs in real time — acting as an intelligent middleware between clients and servers.

---

## 🔐 Features

- ✅ **URL Risk Scoring Engine** – Detects phishing, SQLi, XSS, redirection attacks, and encoded payloads.
- 🔁 **Socket-Based Middleware Service** – Exposes a real-time interface for C/Node/Python proxy clients.
- 🧠 **Explainable Threat Categorization** – Get scores, categories (`SAFE`, `MODERATE RISK`, `DANGEROUS`), and reasons.
- 🔍 **Entropy + Regex + Pattern-Based Detection**
- 🧪 CLI Testing Mode for Generated + Custom URLs
- 📦 Ready for Integration with Future ONNX/CNN-LSTM Models

---

## ⚙️ Architecture

```text
Client ⇄ Proxy (C/Node.js/Python)
            ⇓
      URL Middleware (Python)
         ↳ validate_url()
         ↳ Return JSON response
            ⇓
   Decision: Allow / Block / Alert
