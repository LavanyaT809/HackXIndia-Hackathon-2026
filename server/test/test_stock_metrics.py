
from analytics.stock_loader import StockDataLoader
from analytics.stock_metrics import StockAnalytics

# Load stocks
loader = StockDataLoader()
loader.load_all_stocks()

# Pick a NON-empty stock
symbols = loader.list_stocks()

for symbol in symbols:
    df = loader.get_stock(symbol)
    if not df.empty:
        print("Testing:", symbol)
        analytics = StockAnalytics(df)
        print(analytics.summary())
        break
