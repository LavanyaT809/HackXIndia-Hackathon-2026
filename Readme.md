# 📈 BullBear AI  
### AI-Powered Stock Market Analysis Platform  
**HackXIndia Hackathon 2026**

> 🚧 **Active Development**  
> This project is being built as part of **HackXIndia Hackathon 2026**.  
> Features and implementation are being added iteratively.

---

## 👤 Team Details
- **Team Name:** INNOVATEX (Solo)
- **Participant:** Lavanya

---
## 🎥 Demo Video

📽️ **Project Walkthrough & Demo:**  
👉 [Watch the BullBear AI Demo Video](https://drive.google.com/drive/folders/1bFjOkoT99ZmCDYTcyGNa17-CgOaNZ7yF)

This video demonstrates:
- Homepage & navigation flow
- Stock search and dashboard
- AI-based price prediction (LSTM)
- Portfolio analytics (CAGR, volatility, drawdown, correlation)
- Overall system architecture and features

## 🧠 Project Overview

**BullBear AI** is a full-stack stock market analytics platform that combines real-time stock data, interactive dashboards, and AI/ML-driven insights such as next-day price prediction, market sentiment analysis, and risk indicators.

The project demonstrates the practical application of **machine learning, data engineering, and full-stack development** in a real-world financial use case.

---

## 🚀 Features

### 🔍 Stock Search & Interactive Dashboard
- Search any U.S. stock (AAPL, MSFT, TSLA, etc.)
- Dedicated dashboard per stock
- Live & historical price visualization
- Interactive charts for trend analysis

---

### 🤖 AI Price Prediction (LSTM)
- Predicts **next-day closing price**
- Trained using historical **OHLC data**
- Uses **LSTM (Long Short-Term Memory)** neural networks
- Displays:
  - Predicted price
  - Confidence score based on prediction error

---

### 📰 Market News & Sentiment Analysis
- Stock-related news aggregation
- Sentiment classification:
  - 🟢 Bullish
  - 🔴 Bearish
  - ⚪ Neutral

---

### 🔥 Trending Stocks
- Top gainers
- Top losers
- Most active stocks
- Backend caching for improved performance

---

### ⚠️ Planned AI Enhancements
- Volatility-based risk analysis
- Anomaly detection (price & volume spikes)
- AI-generated stock summaries
- Portfolio-level analytics

---

## 🧠 AI Price Prediction – How It Works

### Model
- **LSTM (Deep Learning – Time Series Forecasting)**
- **Input:** Last *N* days of OHLC data
- **Output:** Next trading day closing price

### Workflow
1. Fetch historical stock market data
2. Normalize data using `MinMaxScaler`
3. Create rolling time-series sequences
4. Train LSTM neural network
5. Evaluate using RMSE and MAE
6. Save trained model
7. Predict next-day price using latest market data

📌 **Dynamic Prediction**
- If today is Aug 21 → predicts Aug 22
- No hard-coded dates

---

## 🧱 Tech Stack

### Frontend
- React.js
- JavaScript
- CSS
- Chart.js / Recharts
- React Router

### Backend
- Python
- Flask
- REST APIs

### Machine Learning
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- LSTM Neural Networks

### Data Sources
- Yahoo Finance (`yfinance`)
- Alpha Vantage API
- Financial Modeling Prep API
- NewsAPI

---

## 📁 Project Structure

HackXIndia-Hackathon-2026/
│

├── client/ # React frontend

│ ├── pages/

│ ├── components/

│ └── App.jsx

│
├── server/

│ ├── app.py # Flask backend

│ ├── ml/

│ │ ├── train_lstm.py # Model training

│ │ └── predict.py # Prediction logic

│ ├── data/ # CSV datasets (ignored in Git)

│ ├── models/ # Saved ML models

│ └── api/

│
├── .env.example # Environment variables template

├── .gitignore

└── README.md



---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/LavanyaT809/HackXIndia-Hackathon-2026.git
cd HackXIndia-Hackathon-2026

cd server
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

ALPHA_VANTAGE_KEY=your_key
FMP_API_KEY=your_key
NEWS_API_KEY=your_key

python app.py

3️⃣ Frontend Setup
cd client
npm install
npm run dev

📊 Model Training (Optional)
cd server/ml
python train_lstm.py

🎥 Demo Video

Demo link will be added before final submission.


---
