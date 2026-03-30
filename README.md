# Telecom-network-failure-prediction
Machine learning system for predicting telecom network failures using Random Forest, FastAPI, and Streamlit for real-time monitoring and predictive maintenance.
# 📡 Telecom Network Failure Prediction System (KSA Use Case)

## 🚀 Overview
This project predicts telecom network failures using Machine Learning.

It simulates telecom-like operating conditions and supports predictive maintenance workflows.

---

## 🎯 Problem Statement
Telecom networks face:
- Signal degradation
- High load
- Environmental effects

👉 Goal: predict failures before they happen.

---

## 🧠 Solution
- ML model (RandomForestClassifier)
- FastAPI backend (/predict endpoint)
- Streamlit dashboard for interactive testing

---

## 📁 Project Structure

telecom-failure-prediction/
│
├── app/
│   ├── main.py
│   └── dashboard.py
│
├── model/
│   ├── train_model.py
│   └── model.pkl          # generated after training
│
├── data/
├── assets/
│   └── dashboard.png      # add your screenshot
│
├── requirements.txt
├── README.md
└── .gitignore


---


## ⚙️ Installation

git clone https://github.com/irfankhalid821-stack
cd telecom-failure-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


---

## ▶️ Run Project

### 1) Train model
python model/train_model.py


### 2) Run API
uvicorn app.main:app --reload


### 3) Run dashboard
streamlit run app/dashboard.py


---

## 📊 API Usage

### Health check
curl http://127.0.0.1:8000/health


### Predict
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "signal_strength": -85,
    "temperature": 40,
    "humidity": 60,
    "network_load": 75
  }'


Expected response:

{
  "prediction": 1
}


---

## 🤖 Model Performance
The training script prints validation metrics (accuracy and F1 score) each time you train.

---


## 👤 Author
Irfan Khalid

- LinkedIn: https://www.linkedin.com/in/irfan-khalid-muhamad-khalid-8b0679130
- GitHub: https://github.com/irfankhalid821-stack
