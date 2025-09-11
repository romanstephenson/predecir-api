# Breast Cancer Recurrence Prediction API

This FastAPI-based machine learning API predicts breast cancer recurrence based on structured clinical features. It uses a versioned model architecture to support future model upgrades without breaking compatibility.

## 🚀 Features

- 🔐 JWT-secured endpoints
- 📊 XGBoost-based recurrence classification
- 🔄 Model versioning via `/predecir/predict` and `/predecir/meta`
- 🧠 Input validation with categorical encoders
- 📤 Metadata endpoint to expose model expectations
- 📦 Auto-token refresh Postman collection

## 📁 Project Structure

```
predecir-api/
├── main.py
├── model/
│   └── v1/
│       ├── model.pkl
│       └── encoders/
├── routers/
│   └── v1/
│       └── prediction.py
├── entity/
│   └── recurrence_input.py
├── auth/
│   └── auth_handler.py
├── utils/
│   └── logging_config.py
├── core/
│   └── config.py
├── postman/
│   └── BreastCancerRecurrenceAPI_v1.postman_collection.json
└── .env.example
```

## 🧪 Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

> Set up a `.env` file using `.env.example`.

## 🔑 Authentication

- Use `/token` to get JWT token (username/password in env)
- Pass token in `Authorization: Bearer <token>` header

## 📬 Endpoints

- `POST /predecir/predict` - Make a prediction
- `GET /predecir/meta` - Get valid field values
- `POST /predecir/token` - Get access token

## 📦 Postman

Import the Postman collection from `/postman` folder. Includes auto-refreshing token setup.

---

## 📄 License
MIT License (c) 2025 Carealytica