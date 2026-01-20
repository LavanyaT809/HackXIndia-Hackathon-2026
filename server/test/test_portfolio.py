from analytics.stock_loader import StockDataLoader
from analytics.portfolio_analytics import PortfolioAnalytics

loader = StockDataLoader()
loader.load_all_stocks()

portfolio = {
    "AAAA_PRICES": 0.3,
    "AAAU_PRICES": 0.4,
    "AAA_PRICES": 0.3
}

pa = PortfolioAnalytics(loader, portfolio)
print(pa.summary())
