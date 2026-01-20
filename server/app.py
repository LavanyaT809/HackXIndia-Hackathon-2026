import threading
import time
import json
import joblib
from explainability import get_shap_explanation
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import pandas as pdv
from routes.portfolio_analysis import portfolio_analysis_bp

# Import Keras model loader and numpy
from tensorflow.keras.models import load_model
import numpy as np 


# 🔐 Load environment variables
load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
FMP_API_KEY = os.getenv('FMP_API_KEY')
NEWS_API_KEY = os.getenv('VITE_NEWS_API_KEY')


app = Flask(__name__)
CORS(app)
# ----------------- DATABASE IMPORTS -----------------
from database.database import Base, engine, SessionLocal
from database.models import Portfolio

# ✅ Ensure tables exist
Base.metadata.create_all(bind=engine)

# ----------------- PORTFOLIO ANALYTICS -----------------
from analytics.portfolio_contribution import portfolio_contribution
from analytics.stock_loader import StockDataLoader



from routes.portfolio import portfolio_bp
app.register_blueprint(portfolio_bp)
app.register_blueprint(portfolio_analysis_bp)

from routes.scatter import scatter_bp
app.register_blueprint(scatter_bp)

from routes.correlation import correlation_bp
app.register_blueprint(correlation_bp)


print("="*60)
print("🔐 API KEYS LOADED:")
print(f"ALPHA_VANTAGE_KEY: {'✅ ' + ALPHA_VANTAGE_KEY[:8] + '...' if ALPHA_VANTAGE_KEY else '❌ Missing'}")
print(f"FMP_API_KEY: {'✅ ' + FMP_API_KEY[:8] + '...' if FMP_API_KEY else '❌ Missing'}")
print(f"NEWS_API_KEY: {'✅ ' + NEWS_API_KEY[:8] + '...' if NEWS_API_KEY else '❌ Missing'}")
print("="*60)



# ----------------- RATE LIMITING & CACHING SYSTEM -----------------


# API Usage Tracking
api_usage = {
    "daily_calls": 0,
    "last_reset": datetime.now().date(),
    "max_daily_calls": 200,  # Conservative limit (FMP free = 250/month ≈ 8/day)
    "calls_today": []
}


# Advanced Caching System
cache_system = {
    "trending": {
        "data": [],
        "last_updated": None,
        "cache_duration": timedelta(hours=2),  # Cache for 2 hours
        "last_api_call": None
    },
    "file_cache_path": "trending_cache.json"  # Persistent file cache
}


def load_file_cache():
    """Load cache from file to survive server restarts"""
    try:
        if os.path.exists(cache_system["file_cache_path"]):
            with open(cache_system["file_cache_path"], 'r') as f:
                file_data = json.load(f)
                if file_data.get("trending"):
                    cache_system["trending"]["data"] = file_data["trending"]["data"]
                    cache_system["trending"]["last_updated"] = datetime.fromisoformat(file_data["trending"]["last_updated"])
                    print(f"✅ Loaded {len(cache_system['trending']['data'])} stocks from file cache")
                    print(f"  Cache date: {cache_system['trending']['last_updated']}")
    except Exception as e:
        print(f"⚠️ Could not load file cache: {e}")


def save_file_cache():
    """Save cache to file"""
    try:
        cache_data = {
            "trending": {
                "data": cache_system["trending"]["data"],
                "last_updated": cache_system["trending"]["last_updated"].isoformat() if cache_system["trending"]["last_updated"] else None
            }
        }
        with open(cache_system["file_cache_path"], 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")


def reset_daily_usage():
    """Reset API usage counter if it's a new day"""
    today = datetime.now().date()
    if api_usage["last_reset"] != today:
        api_usage["daily_calls"] = 0
        api_usage["last_reset"] = today
        api_usage["calls_today"] = []
        print(f"🔄 API usage reset for new day: {today}")


def can_make_api_call():
    """Check if we can make an API call without hitting limits"""
    reset_daily_usage()
    
    # Check daily limit
    if api_usage["daily_calls"] >= api_usage["max_daily_calls"]:
        return False
    
    # Check rate limiting (max 1 call per minute)
    now = datetime.now()
    recent_calls = [call for call in api_usage["calls_today"] if now - call < timedelta(minutes=1)]
    if len(recent_calls) >= 1:
        return False
    
    return True


def track_api_call():
    """Track an API call"""
    api_usage["daily_calls"] += 1
    api_usage["calls_today"].append(datetime.now())
    print(f"📊 API Call logged: {api_usage['daily_calls']}/{api_usage['max_daily_calls']} today")


def is_cache_fresh():
    """Check if cache is still fresh"""
    if not cache_system["trending"]["last_updated"]:
        return False
    
    age = datetime.now() - cache_system["trending"]["last_updated"]
    return age < cache_system["trending"]["cache_duration"]


def safe_fmp_call(endpoint):
    """Make FMP API call with rate limiting and error handling"""
    if not FMP_API_KEY:
        return None
    
    if not can_make_api_call():
        return None
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/{endpoint}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=15)
        track_api_call()
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data
            else:
                return None
        elif response.status_code == 429:
            return None
        else:
            return None
    except Exception:
        return None


def get_mock_trending_data():
    """High-quality mock data for when API is unavailable"""
    return [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "price": 195.25,
            "change": 3.45,
            "changesPercentage": 1.80,
            "volume": 45678900,
            "logo": "https://financialmodelingprep.com/image-stock/AAPL.png"
        },
        {
            "symbol": "MSFT", 
            "name": "Microsoft Corporation",
            "price": 422.80,
            "change": -2.25,
            "changesPercentage": -0.53,
            "volume": 23456789,
            "logo": "https://financialmodelingprep.com/image-stock/MSFT.png"
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "price": 142.30,
            "change": 1.75,
            "changesPercentage": 1.25,
            "volume": 31234567,
            "logo": "https://financialmodelingprep.com/image-stock/GOOGL.png"
        },
        {
            "symbol": "TSLA",
            "name": "Tesla Inc.",
            "price": 248.50,
            "change": 8.20,
            "changesPercentage": 3.41,
            "volume": 89765432,
            "logo": "https://financialmodelingprep.com/image-stock/TSLA.png"
        },
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation", 
            "price": 875.30,
            "change": 15.60,
            "changesPercentage": 1.81,
            "volume": 67543210,
            "logo": "https://financialmodelingprep.com/image-stock/NVDA.png"
        },
        {
            "symbol": "AMZN",
            "name": "Amazon.com Inc.",
            "price": 145.20,
            "change": 2.10,
            "changesPercentage": 1.47,
            "volume": 43218765,
            "logo": "https://financialmodelingprep.com/image-stock/AMZN.png"
        },
        {
            "symbol": "META",
            "name": "Meta Platforms Inc.",
            "price": 312.45,
            "change": -1.80,
            "changesPercentage": -0.57,
            "volume": 28765432,
            "logo": "https://financialmodelingprep.com/image-stock/META.png"
        },
        {
            "symbol": "NFLX",
            "name": "Netflix Inc.",
            "price": 487.90,
            "change": 12.30,
            "changesPercentage": 2.59,
            "volume": 15432189,
            "logo": "https://financialmodelingprep.com/image-stock/NFLX.png"
        },
        {
            "symbol": "CRM",
            "name": "Salesforce Inc.",
            "price": 234.60,
            "change": 4.20,
            "changesPercentage": 1.82,
            "volume": 12876543,
            "logo": "https://financialmodelingprep.com/image-stock/CRM.png"
        },
        {
            "symbol": "AMD",
            "name": "Advanced Micro Devices",
            "price": 134.75,
            "change": 3.85,
            "changesPercentage": 2.94,
            "volume": 34567890,
            "logo": "https://financialmodelingprep.com/image-stock/AMD.png"
        }
    ]


def fetch_trending_data():
    """Smart trending data fetcher with caching and rate limiting"""
    print("\n" + "="*60)
    print("🔄 SMART TRENDING FETCH")
    print("="*60)
    
    # Check if cache is still fresh
    if is_cache_fresh() and cache_system["trending"]["data"]:
        cache_age = datetime.now() - cache_system["trending"]["last_updated"]
        print(f"✅ Using fresh cache (age: {cache_age})")
        print(f"  Cache has {len(cache_system['trending']['data'])} stocks")
        return
    
    # Check API limits before making calls
    reset_daily_usage()
    print(f"📊 API Usage: {api_usage['daily_calls']}/{api_usage['max_daily_calls']} calls today")
    
    if not can_make_api_call():
        print("⛔ Cannot make API calls - using existing cache or mock data")
        if not cache_system["trending"]["data"]:
            print("📝 No cache available - using mock data")
            cache_system["trending"]["data"] = get_mock_trending_data()
            cache_system["trending"]["last_updated"] = datetime.now()
            save_file_cache()
        return
    
    # Try to fetch real data
    print("🌐 Attempting to fetch real data...")
    gainers = safe_fmp_call("stock_market/gainers")
    
    if gainers:
        print("✅ Real data fetched successfully")
        stocks = []
        for stock in gainers[:12]:  # Limit to 12 stocks
            symbol = stock.get("symbol", "").strip()
            if symbol and len(symbol) <= 5:
                try:
                    stocks.append({
                        "symbol": symbol,
                        "name": (stock.get("name") or symbol)[:50],
                        "price": round(float(stock.get("price", 0)), 2),
                        "change": round(float(stock.get("change", 0)), 2),
                        "changesPercentage": round(float(stock.get("changesPercentage", 0)), 2),
                        "volume": int(stock.get("volume", 0)) if stock.get("volume") else 0,
                        "logo": f"https://financialmodelingprep.com/image-stock/{symbol}.png"
                    })
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Skipping {symbol}: data error - {e}")
                    continue
        
        if stocks:
            cache_system["trending"]["data"] = stocks
            cache_system["trending"]["last_updated"] = datetime.now()
            save_file_cache()
            print(f"✅ Cached {len(stocks)} real stocks")
        else:
            print("⚠️ No valid stocks processed - using mock data")
            cache_system["trending"]["data"] = get_mock_trending_data()
            cache_system["trending"]["last_updated"] = datetime.now()
    else:
        print("⚠️ API call failed - using cache or mock data")
        if not cache_system["trending"]["data"]:
            cache_system["trending"]["data"] = get_mock_trending_data()
            cache_system["trending"]["last_updated"] = datetime.now()
            save_file_cache()
    
    print("="*60 + "\n")


# ----------------- STOCK DATA (existing code stays same) -----------------
INTERVAL_TO_FUNCTION = {
    "1d": ("TIME_SERIES_INTRADAY", "60min"),
    "5d": ("TIME_SERIES_DAILY", None),
    "1mo": ("TIME_SERIES_DAILY", None),
    "3mo": ("TIME_SERIES_DAILY", None),
    "6mo": ("TIME_SERIES_DAILY", None),
    "1y": ("TIME_SERIES_DAILY", None)
}


@app.route("/")
def index():
    cache_age = "Never" if not cache_system["trending"]["last_updated"] else str(datetime.now() - cache_system["trending"]["last_updated"])
    return f"""
    <h2>✅ Bull Bear AI Backend</h2>
    <h3>📊 API Status:</h3>
    <ul>
        <li>Daily API calls: {api_usage['daily_calls']}/{api_usage['max_daily_calls']}</li>
        <li>Trending cache: {len(cache_system['trending']['data'])} stocks</li>
        <li>Cache age: {cache_age}</li>
    </ul>
    <h3>🔗 Endpoints:</h3>
    <ul>
        <li><a href="/api/trending">/api/trending</a> - Trending stocks</li>
        <li><a href="/api/status">/api/status</a> - System status</li>
        <li>/api/stock/&lt;symbol&gt; - Individual stock data</li>
        <li>/api/news - News feed</li>
        <li>/api/predict/&lt;symbol&gt; - Stock prediction (POST, input JSON)</li>
    </ul>
    """


@app.route("/api/status")
def get_status():
    """System status endpoint"""
    cache_age = None
    if cache_system["trending"]["last_updated"]:
        cache_age = str(datetime.now() - cache_system["trending"]["last_updated"])
    
    return jsonify({
        "api_usage": api_usage,
        "cache_stats": {
            "trending_stocks": len(cache_system["trending"]["data"]),
            "cache_age": cache_age,
            "is_fresh": is_cache_fresh()
        },
        "system_time": datetime.now().isoformat()
    })


@app.route("/api/stock/<symbol>")
def get_stock(symbol):
    range_param = request.args.get("range", "30d")
    function, interval = INTERVAL_TO_FUNCTION.get(range_param, ("TIME_SERIES_DAILY", None))

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_KEY,
    }
    if interval:
        params["interval"] = interval

    try:
        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
        data = resp.json()

        key = next((k for k in data if "Time Series" in k), None)
        if not key or not isinstance(data[key], dict):
            return jsonify({"error": data.get("Note") or data.get("Error Message") or "Data not available."}), 404

        day_counts = {
            "1d": 1, "5d": 5, "1mo": 22, "3mo": 63,
            "6mo": 126, "1y": 252, "30d": 30,
        }
        days = day_counts.get(range_param, 30)
        rows = []
        for d_str in sorted(data[key].keys(), reverse=False)[-days:]:
            day = data[key][d_str]
            rows.append({
                "date": d_str,
                "open": float(day["1. open"]),
                "high": float(day["2. high"]),
                "low": float(day["3. low"]),
                "close": float(day["4. close"]),
                "volume": int(day["5. volume"])
            })

        latest = rows[-1]
        return jsonify({
            "symbol": symbol,
            "current_price": latest["close"],
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "volume": latest["volume"],
            "last_updated": latest["date"] + " " + datetime.now().strftime('%H:%M:%S'),
            "historical": rows,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------- TRENDING API -----------------


@app.route("/api/trending")
def get_trending():
    """Get trending stocks - guaranteed to work, never hits rate limits"""
    print(f"\n🔍 TRENDING REQUEST")
    
    try:
        # Always serve from cache if available
        if cache_system["trending"]["data"]:
            cache_age = datetime.now() - cache_system["trending"]["last_updated"] if cache_system["trending"]["last_updated"] else "Unknown"
            is_fresh = is_cache_fresh()
            
            print(f"  Serving from cache: {len(cache_system['trending']['data'])} stocks")
            print(f"  Cache age: {cache_age}")
            print(f"  Is fresh: {is_fresh}")
            
            # If cache is stale and we can make API calls, trigger background refresh
            if not is_fresh and can_make_api_call():
                print("  🔄 Triggering background refresh...")
                threading.Thread(target=fetch_trending_data, daemon=True).start()
            
            response = {
                "last_updated": cache_system["trending"]["last_updated"].strftime('%Y-%m-%d %H:%M:%S') if cache_system["trending"]["last_updated"] else "Mock data",
                "stocks": cache_system["trending"]["data"],
                "cache_info": {
                    "is_fresh": is_fresh,
                    "age_minutes": int((datetime.now() - cache_system["trending"]["last_updated"]).total_seconds() / 60) if cache_system["trending"]["last_updated"] else 0,
                    "api_calls_today": api_usage["daily_calls"],
                    "source": "real_data" if cache_system["trending"]["last_updated"] and "(Mock)" not in str(cache_system["trending"]["last_updated"]) else "mock_data"
                }
            }
            
            print(f"📤 Response sent: {len(response['stocks'])} stocks")
            return jsonify(response)
        
        else:
            # No cache available - try to fetch or use mock
            print("  No cache available")
            if can_make_api_call():
                print("  Fetching fresh data...")
                fetch_trending_data()
            else:
                print("  Using mock data (no API calls available)")
                cache_system["trending"]["data"] = get_mock_trending_data()
                cache_system["trending"]["last_updated"] = datetime.now()
            
            response = {
                "last_updated": cache_system["trending"]["last_updated"].strftime('%Y-%m-%d %H:%M:%S') if cache_system["trending"]["last_updated"] else "Mock data",
                "stocks": cache_system["trending"]["data"]
            }
            
            return jsonify(response)
            
    except Exception as e:
        print(f"❌ Critical error in trending route: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency fallback - always works
        return jsonify({
            "last_updated": "Emergency fallback",
            "stocks": get_mock_trending_data()[:5],  # Just 5 stocks for emergency
            "error": "Server error, using fallback data"
        })


        
# ----------------- NEWS API (existing code) -----------------
news_cache = {}
NEWS_CACHE_TTL = timedelta(minutes=30)


@app.route("/api/news")
def get_news():
    query = request.args.get("q", "stock market").strip().lower()
    sort_by = request.args.get("sortBy", "publishedAt")
    page_size = request.args.get("pageSize", 30)

    cache_key = f"{query}_{sort_by}_{page_size}"
    now = datetime.now()

    if cache_key in news_cache:
        cached = news_cache[cache_key]
        if now - cached["time"] < NEWS_CACHE_TTL:
            return jsonify(cached["data"])

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": sort_by,
                "pageSize": page_size,
                "apiKey": NEWS_API_KEY,
                "language": "en",
            },
            timeout=10
        )
        data = resp.json()
        news_cache[cache_key] = {"data": data, "time": now}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------- MODEL LOADING AND PREDICTION API -----------------
MODEL_DIR = "ml/models"  # Directory where your models and scalers are saved

def load_stock_model(symbol):
    """Load Keras model for a given stock symbol"""
    model_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_lstm_model.h5")
    if not os.path.exists(model_path):
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
        return None


def load_stock_scaler(symbol):
    """Load scaler object for a given stock symbol"""
    scaler_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_scaler.save")
    if not os.path.exists(scaler_path):
        return None
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        print(f"Error loading scaler for {symbol}: {e}")
        return None


@app.route("/api/predict/<symbol>", methods=["POST"])
def predict_stock(symbol):
    """
    Predict stock closing price for given symbol using the pre-trained LSTM model.

    Also returns per-feature per-timestep SHAP explainability values.
    """
    try:
        data = request.get_json(force=True)
        raw_features = data.get("features")
        if raw_features is None:
            return jsonify({"error": "Missing 'features' key in JSON body."}), 400
        
        raw_array = np.array(raw_features, dtype=np.float32)
        
        if raw_array.shape != (10, 4):
            return jsonify({"error": f"Input 'features' must be shape (10, 4), got {raw_array.shape}"}), 400
        
        scaler = load_stock_scaler(symbol)
        model = load_stock_model(symbol)
        
        if scaler is None or model is None:
            return jsonify({"error": f"Model or scaler for symbol '{symbol}' not found."}), 404
        
        scaled_features = scaler.transform(raw_array)
        input_seq = scaled_features.reshape(1, 10, 4)

        pred_scaled = model.predict(input_seq)
        close_index = 0  # Close price is the first feature
        data_min = scaler.data_min_[close_index]
        data_max = scaler.data_max_[close_index]
        pred_real_close = pred_scaled[0][0] * (data_max - data_min) + data_min

        shap_array = get_shap_explanation(model, input_seq)  # shape (1, 10, 4)

        # 1️⃣ Aggregate SHAP values: mean over timesteps
        feature_importance = np.mean(np.abs(shap_array[0]), axis=0)  # shape (4,)

        # 2️⃣ Normalize to percentage
        total_importance = np.sum(feature_importance)

        if total_importance > 0:
            feature_importance = (feature_importance / total_importance) * 100
        else:
            feature_importance = np.zeros(4)

        # 3️⃣ Convert to Python list (ONLY at the end)
        feature_importance = np.round(feature_importance, 2).tolist()


 
        # SHAP values per timestep, per feature
        shap_per_timestep = np.round(shap_array[0], 4).tolist()  # shape (10,4)
        feature_names = ['Close', 'MA10', 'MA50', 'Return']
        print("SHAP Array sample:", shap_array[0][:2])  # print first 2 timesteps x features
        print("Feature Importance:", feature_importance)
        print("Shape of timestep details:", np.array(shap_array[0]).shape)



        return jsonify({
            "symbol": symbol.upper(),
            "predicted_close_price": round(float(pred_real_close), 2),
            "explainability": {
                "feature_names": feature_names,
                "global_importance": feature_importance,     # [importance for each feature]
                "timestep_details": shap_per_timestep        # [ [shap0, shap1, ...], ... (10 timesteps)]
            }
        })
    except Exception as e:
        print(f"Prediction error for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500



def intelligent_background_updater():
    """Smart background updater that respects rate limits"""
    print("🤖 Intelligent updater started")
    while True:
        try:
            time.sleep(4 * 60 * 60)  # 4 hours
            print("⏰ Background update check...")
            if not is_cache_fresh() and can_make_api_call():
                print("🔄 Background fetch triggered")
                fetch_trending_data()
            else:
                if is_cache_fresh():
                    print("✅ Cache still fresh - skipping update")
                else:
                    print("⛔ Cache stale but no API calls available")
        except Exception as e:
            print(f"❌ Background updater error: {e}")
            time.sleep(60)  # Wait 1 minute before retry


threading.Thread(target=intelligent_background_updater, daemon=True).start()




@app.route("/api/portfolio/contribution/<int:user_id>")
def portfolio_contribution_api(user_id):
    db = SessionLocal()

    rows = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()

    if not rows:
        return jsonify({"error": "No portfolio found"}), 404

    portfolio_df = pd.DataFrame([
        {"symbol": r.symbol, "weight": r.weight} for r in rows
    ])

    loader = StockDataLoader()
    loader.load_all_stocks()

    stock_data_map = {}
    for symbol in portfolio_df["symbol"]:
        df = loader.get_stock(symbol)
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404
        stock_data_map[symbol] = df

    result = portfolio_contribution(portfolio_df, stock_data_map)
    return jsonify(result)



# ----------------- STARTUP -----------------
if __name__ == "__main__":
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    print("\n🚀 BULL BEAR AI - RATE LIMIT PROTECTED")
    print("="*60)
    load_file_cache()
    reset_daily_usage()

    if not FMP_API_KEY:
        print("❌ CRITICAL: No FMP API key!")
        print("  Will use mock data only")

    if can_make_api_call() and not is_cache_fresh():
        print("⚡ Initial fetch...")
        fetch_trending_data()
    else:
        if is_cache_fresh():
            print("✅ Using existing fresh cache")
        else:
            print("⚡ Using mock data (preserving API quota)")
            cache_system["trending"]["data"] = get_mock_trending_data()
            cache_system["trending"]["last_updated"] = datetime.now()

    print(f"\n🌟 Server ready!")
    print(f"📊 Trending stocks: {len(cache_system['trending']['data'])}")
    print(f"📊 API calls remaining today: {api_usage['max_daily_calls'] - api_usage['daily_calls']}")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)














# import threading
# import time
# import json
# import joblib
# from explainability import get_shap_explanation
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import requests
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import os
# import pandas as pdv
# from routes.portfolio_analysis import portfolio_analysis_bp

# # Import Keras model loader and numpy
# from tensorflow.keras.models import load_model
# import numpy as np 


# # 🔐 Load environment variables
# load_dotenv()
# ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
# FMP_API_KEY = os.getenv('FMP_API_KEY')
# NEWS_API_KEY = os.getenv('VITE_NEWS_API_KEY')


# app = Flask(__name__)
# CORS(app)
# # ----------------- DATABASE IMPORTS -----------------
# from database.database import Base, engine, SessionLocal
# from database.models import Portfolio

# # ✅ Ensure tables exist
# Base.metadata.create_all(bind=engine)

# # ----------------- PORTFOLIO ANALYTICS -----------------
# from analytics.portfolio_contribution import portfolio_contribution
# from analytics.stock_loader import StockDataLoader



# from routes.portfolio import portfolio_bp
# app.register_blueprint(portfolio_bp)
# app.register_blueprint(portfolio_analysis_bp)

# from routes.scatter import scatter_bp
# app.register_blueprint(scatter_bp)

# from routes.correlation import correlation_bp
# app.register_blueprint(correlation_bp)


# print("="*60)
# print("🔐 API KEYS LOADED:")
# print(f"ALPHA_VANTAGE_KEY: {'✅ ' + ALPHA_VANTAGE_KEY[:8] + '...' if ALPHA_VANTAGE_KEY else '❌ Missing'}")
# print(f"FMP_API_KEY: {'✅ ' + FMP_API_KEY[:8] + '...' if FMP_API_KEY else '❌ Missing'}")
# print(f"NEWS_API_KEY: {'✅ ' + NEWS_API_KEY[:8] + '...' if NEWS_API_KEY else '❌ Missing'}")
# print("="*60)



# # ----------------- RATE LIMITING & CACHING SYSTEM -----------------


# # API Usage Tracking
# api_usage = {
#     "daily_calls": 0,
#     "last_reset": datetime.now().date(),
#     "max_daily_calls": 200,  # Conservative limit (FMP free = 250/month ≈ 8/day)
#     "calls_today": []
# }


# # Advanced Caching System
# cache_system = {
#     "trending": {
#         "data": [],
#         "last_updated": None,
#         "cache_duration": timedelta(hours=2),  # Cache for 2 hours
#         "last_api_call": None
#     },
#     "file_cache_path": "trending_cache.json"  # Persistent file cache
# }


# def load_file_cache():
#     """Load cache from file to survive server restarts"""
#     try:
#         if os.path.exists(cache_system["file_cache_path"]):
#             with open(cache_system["file_cache_path"], 'r') as f:
#                 file_data = json.load(f)
#                 if file_data.get("trending"):
#                     cache_system["trending"]["data"] = file_data["trending"]["data"]
#                     cache_system["trending"]["last_updated"] = datetime.fromisoformat(file_data["trending"]["last_updated"])
#                     print(f"✅ Loaded {len(cache_system['trending']['data'])} stocks from file cache")
#                     print(f"  Cache date: {cache_system['trending']['last_updated']}")
#     except Exception as e:
#         print(f"⚠️ Could not load file cache: {e}")


# def save_file_cache():
#     """Save cache to file"""
#     try:
#         cache_data = {
#             "trending": {
#                 "data": cache_system["trending"]["data"],
#                 "last_updated": cache_system["trending"]["last_updated"].isoformat() if cache_system["trending"]["last_updated"] else None
#             }
#         }
#         with open(cache_system["file_cache_path"], 'w') as f:
#             json.dump(cache_data, f, indent=2)
#     except Exception as e:
#         print(f"⚠️ Could not save cache: {e}")


# def reset_daily_usage():
#     """Reset API usage counter if it's a new day"""
#     today = datetime.now().date()
#     if api_usage["last_reset"] != today:
#         api_usage["daily_calls"] = 0
#         api_usage["last_reset"] = today
#         api_usage["calls_today"] = []
#         print(f"🔄 API usage reset for new day: {today}")


# def can_make_api_call():
#     """Check if we can make an API call without hitting limits"""
#     reset_daily_usage()
    
#     # Check daily limit
#     if api_usage["daily_calls"] >= api_usage["max_daily_calls"]:
#         return False
    
#     # Check rate limiting (max 1 call per minute)
#     now = datetime.now()
#     recent_calls = [call for call in api_usage["calls_today"] if now - call < timedelta(minutes=1)]
#     if len(recent_calls) >= 1:
#         return False
    
#     return True


# def track_api_call():
#     """Track an API call"""
#     api_usage["daily_calls"] += 1
#     api_usage["calls_today"].append(datetime.now())
#     print(f"📊 API Call logged: {api_usage['daily_calls']}/{api_usage['max_daily_calls']} today")


# def is_cache_fresh():
#     """Check if cache is still fresh"""
#     if not cache_system["trending"]["last_updated"]:
#         return False
    
#     age = datetime.now() - cache_system["trending"]["last_updated"]
#     return age < cache_system["trending"]["cache_duration"]


# def safe_fmp_call(endpoint):
#     """Make FMP API call with rate limiting and error handling"""
#     if not FMP_API_KEY:
#         return None
    
#     if not can_make_api_call():
#         return None
    
#     try:
#         url = f"https://financialmodelingprep.com/api/v3/{endpoint}?apikey={FMP_API_KEY}"
#         response = requests.get(url, timeout=15)
#         track_api_call()
        
#         if response.status_code == 200:
#             data = response.json()
#             if isinstance(data, list) and len(data) > 0:
#                 return data
#             else:
#                 return None
#         elif response.status_code == 429:
#             return None
#         else:
#             return None
#     except Exception:
#         return None


# def get_mock_trending_data():
#     """High-quality mock data for when API is unavailable"""
#     return [
#         {
#             "symbol": "AAPL",
#             "name": "Apple Inc.",
#             "price": 195.25,
#             "change": 3.45,
#             "changesPercentage": 1.80,
#             "volume": 45678900,
#             "logo": "https://financialmodelingprep.com/image-stock/AAPL.png"
#         },
#         {
#             "symbol": "MSFT", 
#             "name": "Microsoft Corporation",
#             "price": 422.80,
#             "change": -2.25,
#             "changesPercentage": -0.53,
#             "volume": 23456789,
#             "logo": "https://financialmodelingprep.com/image-stock/MSFT.png"
#         },
#         {
#             "symbol": "GOOGL",
#             "name": "Alphabet Inc.",
#             "price": 142.30,
#             "change": 1.75,
#             "changesPercentage": 1.25,
#             "volume": 31234567,
#             "logo": "https://financialmodelingprep.com/image-stock/GOOGL.png"
#         },
#         {
#             "symbol": "TSLA",
#             "name": "Tesla Inc.",
#             "price": 248.50,
#             "change": 8.20,
#             "changesPercentage": 3.41,
#             "volume": 89765432,
#             "logo": "https://financialmodelingprep.com/image-stock/TSLA.png"
#         },
#         {
#             "symbol": "NVDA",
#             "name": "NVIDIA Corporation", 
#             "price": 875.30,
#             "change": 15.60,
#             "changesPercentage": 1.81,
#             "volume": 67543210,
#             "logo": "https://financialmodelingprep.com/image-stock/NVDA.png"
#         },
#         {
#             "symbol": "AMZN",
#             "name": "Amazon.com Inc.",
#             "price": 145.20,
#             "change": 2.10,
#             "changesPercentage": 1.47,
#             "volume": 43218765,
#             "logo": "https://financialmodelingprep.com/image-stock/AMZN.png"
#         },
#         {
#             "symbol": "META",
#             "name": "Meta Platforms Inc.",
#             "price": 312.45,
#             "change": -1.80,
#             "changesPercentage": -0.57,
#             "volume": 28765432,
#             "logo": "https://financialmodelingprep.com/image-stock/META.png"
#         },
#         {
#             "symbol": "NFLX",
#             "name": "Netflix Inc.",
#             "price": 487.90,
#             "change": 12.30,
#             "changesPercentage": 2.59,
#             "volume": 15432189,
#             "logo": "https://financialmodelingprep.com/image-stock/NFLX.png"
#         },
#         {
#             "symbol": "CRM",
#             "name": "Salesforce Inc.",
#             "price": 234.60,
#             "change": 4.20,
#             "changesPercentage": 1.82,
#             "volume": 12876543,
#             "logo": "https://financialmodelingprep.com/image-stock/CRM.png"
#         },
#         {
#             "symbol": "AMD",
#             "name": "Advanced Micro Devices",
#             "price": 134.75,
#             "change": 3.85,
#             "changesPercentage": 2.94,
#             "volume": 34567890,
#             "logo": "https://financialmodelingprep.com/image-stock/AMD.png"
#         }
#     ]


# def fetch_trending_data():
#     """Smart trending data fetcher with caching and rate limiting"""
#     print("\n" + "="*60)
#     print("🔄 SMART TRENDING FETCH")
#     print("="*60)
    
#     # Check if cache is still fresh
#     if is_cache_fresh() and cache_system["trending"]["data"]:
#         cache_age = datetime.now() - cache_system["trending"]["last_updated"]
#         print(f"✅ Using fresh cache (age: {cache_age})")
#         print(f"  Cache has {len(cache_system['trending']['data'])} stocks")
#         return
    
#     # Check API limits before making calls
#     reset_daily_usage()
#     print(f"📊 API Usage: {api_usage['daily_calls']}/{api_usage['max_daily_calls']} calls today")
    
#     if not can_make_api_call():
#         print("⛔ Cannot make API calls - using existing cache or mock data")
#         if not cache_system["trending"]["data"]:
#             print("📝 No cache available - using mock data")
#             cache_system["trending"]["data"] = get_mock_trending_data()
#             cache_system["trending"]["last_updated"] = datetime.now()
#             save_file_cache()
#         return
    
#     # Try to fetch real data
#     print("🌐 Attempting to fetch real data...")
#     gainers = safe_fmp_call("stock_market/gainers")
    
#     if gainers:
#         print("✅ Real data fetched successfully")
#         stocks = []
#         for stock in gainers[:12]:  # Limit to 12 stocks
#             symbol = stock.get("symbol", "").strip()
#             if symbol and len(symbol) <= 5:
#                 try:
#                     stocks.append({
#                         "symbol": symbol,
#                         "name": (stock.get("name") or symbol)[:50],
#                         "price": round(float(stock.get("price", 0)), 2),
#                         "change": round(float(stock.get("change", 0)), 2),
#                         "changesPercentage": round(float(stock.get("changesPercentage", 0)), 2),
#                         "volume": int(stock.get("volume", 0)) if stock.get("volume") else 0,
#                         "logo": f"https://financialmodelingprep.com/image-stock/{symbol}.png"
#                     })
#                 except (ValueError, TypeError) as e:
#                     print(f"⚠️ Skipping {symbol}: data error - {e}")
#                     continue
        
#         if stocks:
#             cache_system["trending"]["data"] = stocks
#             cache_system["trending"]["last_updated"] = datetime.now()
#             save_file_cache()
#             print(f"✅ Cached {len(stocks)} real stocks")
#         else:
#             print("⚠️ No valid stocks processed - using mock data")
#             cache_system["trending"]["data"] = get_mock_trending_data()
#             cache_system["trending"]["last_updated"] = datetime.now()
#     else:
#         print("⚠️ API call failed - using cache or mock data")
#         if not cache_system["trending"]["data"]:
#             cache_system["trending"]["data"] = get_mock_trending_data()
#             cache_system["trending"]["last_updated"] = datetime.now()
#             save_file_cache()
    
#     print("="*60 + "\n")


# # ----------------- STOCK DATA (existing code stays same) -----------------
# INTERVAL_TO_FUNCTION = {
#     "1d": ("TIME_SERIES_INTRADAY", "60min"),
#     "5d": ("TIME_SERIES_DAILY", None),
#     "1mo": ("TIME_SERIES_DAILY", None),
#     "3mo": ("TIME_SERIES_DAILY", None),
#     "6mo": ("TIME_SERIES_DAILY", None),
#     "1y": ("TIME_SERIES_DAILY", None)
# }


# @app.route("/")
# def index():
#     cache_age = "Never" if not cache_system["trending"]["last_updated"] else str(datetime.now() - cache_system["trending"]["last_updated"])
#     return f"""
#     <h2>✅ Bull Bear AI Backend</h2>
#     <h3>📊 API Status:</h3>
#     <ul>
#         <li>Daily API calls: {api_usage['daily_calls']}/{api_usage['max_daily_calls']}</li>
#         <li>Trending cache: {len(cache_system['trending']['data'])} stocks</li>
#         <li>Cache age: {cache_age}</li>
#     </ul>
#     <h3>🔗 Endpoints:</h3>
#     <ul>
#         <li><a href="/api/trending">/api/trending</a> - Trending stocks</li>
#         <li><a href="/api/status">/api/status</a> - System status</li>
#         <li>/api/stock/&lt;symbol&gt; - Individual stock data</li>
#         <li>/api/news - News feed</li>
#         <li>/api/predict/&lt;symbol&gt; - Stock prediction (POST, input JSON)</li>
#     </ul>
#     """


# @app.route("/api/status")
# def get_status():
#     """System status endpoint"""
#     cache_age = None
#     if cache_system["trending"]["last_updated"]:
#         cache_age = str(datetime.now() - cache_system["trending"]["last_updated"])
    
#     return jsonify({
#         "api_usage": api_usage,
#         "cache_stats": {
#             "trending_stocks": len(cache_system["trending"]["data"]),
#             "cache_age": cache_age,
#             "is_fresh": is_cache_fresh()
#         },
#         "system_time": datetime.now().isoformat()
#     })


# @app.route("/api/stock/<symbol>")
# def get_stock(symbol):
#     range_param = request.args.get("range", "30d")
#     function, interval = INTERVAL_TO_FUNCTION.get(range_param, ("TIME_SERIES_DAILY", None))

#     params = {
#         "function": function,
#         "symbol": symbol,
#         "apikey": ALPHA_VANTAGE_KEY,
#     }
#     if interval:
#         params["interval"] = interval

#     try:
#         resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
#         data = resp.json()

#         key = next((k for k in data if "Time Series" in k), None)
#         if not key or not isinstance(data[key], dict):
#             return jsonify({"error": data.get("Note") or data.get("Error Message") or "Data not available."}), 404

#         day_counts = {
#             "1d": 1, "5d": 5, "1mo": 22, "3mo": 63,
#             "6mo": 126, "1y": 252, "30d": 30,
#         }
#         days = day_counts.get(range_param, 30)
#         rows = []
#         for d_str in sorted(data[key].keys(), reverse=False)[-days:]:
#             day = data[key][d_str]
#             rows.append({
#                 "date": d_str,
#                 "open": float(day["1. open"]),
#                 "high": float(day["2. high"]),
#                 "low": float(day["3. low"]),
#                 "close": float(day["4. close"]),
#                 "volume": int(day["5. volume"])
#             })

#         latest = rows[-1]
#         return jsonify({
#             "symbol": symbol,
#             "current_price": latest["close"],
#             "open": latest["open"],
#             "high": latest["high"],
#             "low": latest["low"],
#             "volume": latest["volume"],
#             "last_updated": latest["date"] + " " + datetime.now().strftime('%H:%M:%S'),
#             "historical": rows,
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ----------------- TRENDING API -----------------


# @app.route("/api/trending")
# def get_trending():
#     """Get trending stocks - guaranteed to work, never hits rate limits"""
#     print(f"\n🔍 TRENDING REQUEST")
    
#     try:
#         # Always serve from cache if available
#         if cache_system["trending"]["data"]:
#             cache_age = datetime.now() - cache_system["trending"]["last_updated"] if cache_system["trending"]["last_updated"] else "Unknown"
#             is_fresh = is_cache_fresh()
            
#             print(f"  Serving from cache: {len(cache_system['trending']['data'])} stocks")
#             print(f"  Cache age: {cache_age}")
#             print(f"  Is fresh: {is_fresh}")
            
#             # If cache is stale and we can make API calls, trigger background refresh
#             if not is_fresh and can_make_api_call():
#                 print("  🔄 Triggering background refresh...")
#                 threading.Thread(target=fetch_trending_data, daemon=True).start()
            
#             response = {
#                 "last_updated": cache_system["trending"]["last_updated"].strftime('%Y-%m-%d %H:%M:%S') if cache_system["trending"]["last_updated"] else "Mock data",
#                 "stocks": cache_system["trending"]["data"],
#                 "cache_info": {
#                     "is_fresh": is_fresh,
#                     "age_minutes": int((datetime.now() - cache_system["trending"]["last_updated"]).total_seconds() / 60) if cache_system["trending"]["last_updated"] else 0,
#                     "api_calls_today": api_usage["daily_calls"],
#                     "source": "real_data" if cache_system["trending"]["last_updated"] and "(Mock)" not in str(cache_system["trending"]["last_updated"]) else "mock_data"
#                 }
#             }
            
#             print(f"📤 Response sent: {len(response['stocks'])} stocks")
#             return jsonify(response)
        
#         else:
#             # No cache available - try to fetch or use mock
#             print("  No cache available")
#             if can_make_api_call():
#                 print("  Fetching fresh data...")
#                 fetch_trending_data()
#             else:
#                 print("  Using mock data (no API calls available)")
#                 cache_system["trending"]["data"] = get_mock_trending_data()
#                 cache_system["trending"]["last_updated"] = datetime.now()
            
#             response = {
#                 "last_updated": cache_system["trending"]["last_updated"].strftime('%Y-%m-%d %H:%M:%S') if cache_system["trending"]["last_updated"] else "Mock data",
#                 "stocks": cache_system["trending"]["data"]
#             }
            
#             return jsonify(response)
            
#     except Exception as e:
#         print(f"❌ Critical error in trending route: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Emergency fallback - always works
#         return jsonify({
#             "last_updated": "Emergency fallback",
#             "stocks": get_mock_trending_data()[:5],  # Just 5 stocks for emergency
#             "error": "Server error, using fallback data"
#         })


        
# # ----------------- NEWS API (existing code) -----------------
# news_cache = {}
# NEWS_CACHE_TTL = timedelta(minutes=30)


# @app.route("/api/news")
# def get_news():
#     query = request.args.get("q", "stock market").strip().lower()
#     sort_by = request.args.get("sortBy", "publishedAt")
#     page_size = request.args.get("pageSize", 30)

#     cache_key = f"{query}_{sort_by}_{page_size}"
#     now = datetime.now()

#     if cache_key in news_cache:
#         cached = news_cache[cache_key]
#         if now - cached["time"] < NEWS_CACHE_TTL:
#             return jsonify(cached["data"])

#     try:
#         resp = requests.get(
#             "https://newsapi.org/v2/everything",
#             params={
#                 "q": query,
#                 "sortBy": sort_by,
#                 "pageSize": page_size,
#                 "apiKey": NEWS_API_KEY,
#                 "language": "en",
#             },
#             timeout=10
#         )
#         data = resp.json()
#         news_cache[cache_key] = {"data": data, "time": now}
#         return jsonify(data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ----------------- MODEL LOADING AND PREDICTION API -----------------
# MODEL_DIR = "ml/models"  # Directory where your models and scalers are saved

# def load_stock_model(symbol):
#     """Load Keras model for a given stock symbol"""
#     model_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_lstm_model.h5")
#     if not os.path.exists(model_path):
#         return None
#     try:
#         model = load_model(model_path)
#         return model
#     except Exception as e:
#         print(f"Error loading model for {symbol}: {e}")
#         return None


# def load_stock_scaler(symbol):
#     """Load scaler object for a given stock symbol"""
#     scaler_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_scaler.save")
#     if not os.path.exists(scaler_path):
#         return None
#     try:
#         scaler = joblib.load(scaler_path)
#         return scaler
#     except Exception as e:
#         print(f"Error loading scaler for {symbol}: {e}")
#         return None


# @app.route("/api/predict/<symbol>", methods=["POST"])
# def predict_stock(symbol):
#     """
#     Predict stock closing price for given symbol using the pre-trained LSTM model.

#     Also returns per-feature per-timestep SHAP explainability values.
#     """
#     try:
#         data = request.get_json(force=True)
#         raw_features = data.get("features")
#         if raw_features is None:
#             return jsonify({"error": "Missing 'features' key in JSON body."}), 400
        
#         raw_array = np.array(raw_features, dtype=np.float32)
        
#         if raw_array.shape != (10, 4):
#             return jsonify({"error": f"Input 'features' must be shape (10, 4), got {raw_array.shape}"}), 400
        
#         scaler = load_stock_scaler(symbol)
#         model = load_stock_model(symbol)
        
#         if scaler is None or model is None:
#             return jsonify({"error": f"Model or scaler for symbol '{symbol}' not found."}), 404
        
#         scaled_features = scaler.transform(raw_array)
#         input_seq = scaled_features.reshape(1, 10, 4)

#         pred_scaled = model.predict(input_seq)
#         close_index = 0  # Close price is the first feature
#         data_min = scaler.data_min_[close_index]
#         data_max = scaler.data_max_[close_index]
#         pred_real_close = pred_scaled[0][0] * (data_max - data_min) + data_min

#         shap_array = get_shap_explanation(model, input_seq)  # shape (1, 10, 4)

#         # 1️⃣ Aggregate SHAP values: mean over timesteps
#         feature_importance = np.mean(np.abs(shap_array[0]), axis=0)  # shape (4,)

#         # 2️⃣ Normalize to percentage
#         total_importance = np.sum(feature_importance)

#         if total_importance > 0:
#             feature_importance = (feature_importance / total_importance) * 100
#         else:
#             feature_importance = np.zeros(4)

#         # 3️⃣ Convert to Python list (ONLY at the end)
#         feature_importance = np.round(feature_importance, 2).tolist()


 
#         # SHAP values per timestep, per feature
#         shap_per_timestep = np.round(shap_array[0], 4).tolist()  # shape (10,4)
#         feature_names = ['Close', 'MA10', 'MA50', 'Return']
#         print("SHAP Array sample:", shap_array[0][:2])  # print first 2 timesteps x features
#         print("Feature Importance:", feature_importance)
#         print("Shape of timestep details:", np.array(shap_array[0]).shape)



#         return jsonify({
#             "symbol": symbol.upper(),
#             "predicted_close_price": round(float(pred_real_close), 2),
#             "explainability": {
#                 "feature_names": feature_names,
#                 "global_importance": feature_importance,     # [importance for each feature]
#                 "timestep_details": shap_per_timestep        # [ [shap0, shap1, ...], ... (10 timesteps)]
#             }
#         })
#     except Exception as e:
#         print(f"Prediction error for {symbol}: {e}")
#         return jsonify({"error": str(e)}), 500



# def intelligent_background_updater():
#     """Smart background updater that respects rate limits"""
#     print("🤖 Intelligent updater started")
#     while True:
#         try:
#             time.sleep(4 * 60 * 60)  # 4 hours
#             print("⏰ Background update check...")
#             if not is_cache_fresh() and can_make_api_call():
#                 print("🔄 Background fetch triggered")
#                 fetch_trending_data()
#             else:
#                 if is_cache_fresh():
#                     print("✅ Cache still fresh - skipping update")
#                 else:
#                     print("⛔ Cache stale but no API calls available")
#         except Exception as e:
#             print(f"❌ Background updater error: {e}")
#             time.sleep(60)  # Wait 1 minute before retry


# threading.Thread(target=intelligent_background_updater, daemon=True).start()




# @app.route("/api/portfolio/contribution/<int:user_id>")
# def portfolio_contribution_api(user_id):
#     db = SessionLocal()

#     rows = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()

#     if not rows:
#         return jsonify({"error": "No portfolio found"}), 404

#     portfolio_df = pd.DataFrame([
#         {"symbol": r.symbol, "weight": r.weight} for r in rows
#     ])

#     loader = StockDataLoader()
#     loader.load_all_stocks()

#     stock_data_map = {}
#     for symbol in portfolio_df["symbol"]:
#         df = loader.get_stock(symbol)
#         if df.empty:
#             return jsonify({"error": f"No data for {symbol}"}), 404
#         stock_data_map[symbol] = df

#     result = portfolio_contribution(portfolio_df, stock_data_map)
#     return jsonify(result)



# # ----------------- STARTUP -----------------
# if __name__ == "__main__":
#     # Create DB tables
#     Base.metadata.create_all(bind=engine)
#     print("\n🚀 BULL BEAR AI - RATE LIMIT PROTECTED")
#     print("="*60)
#     load_file_cache()
#     reset_daily_usage()

#     if not FMP_API_KEY:
#         print("❌ CRITICAL: No FMP API key!")
#         print("  Will use mock data only")

#     if can_make_api_call() and not is_cache_fresh():
#         print("⚡ Initial fetch...")
#         fetch_trending_data()
#     else:
#         if is_cache_fresh():
#             print("✅ Using existing fresh cache")
#         else:
#             print("⚡ Using mock data (preserving API quota)")
#             cache_system["trending"]["data"] = get_mock_trending_data()
#             cache_system["trending"]["last_updated"] = datetime.now()

#     print(f"\n🌟 Server ready!")
#     print(f"📊 Trending stocks: {len(cache_system['trending']['data'])}")
#     print(f"📊 API calls remaining today: {api_usage['max_daily_calls'] - api_usage['daily_calls']}")
#     print("="*60)

#     app.run(debug=True, host='0.0.0.0', port=5000)







