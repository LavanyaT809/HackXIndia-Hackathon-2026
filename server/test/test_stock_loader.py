from analytics.stock_loader import StockDataLoader

loader = StockDataLoader()
loader.load_all_stocks()

print("Total stocks loaded:", len(loader.list_stocks()))
print("Sample stock symbols:", loader.list_stocks()[:10])

symbol = loader.list_stocks()[0]
df = loader.get_stock(symbol)

print(f"\nData for {symbol}:")
print(df.head())
print(df.tail())
