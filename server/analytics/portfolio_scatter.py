from analytics.stock_metrics import StockAnalytics
from analytics.stock_loader import StockDataLoader


class PortfolioReturnRiskScatter:
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.loader = StockDataLoader("data_cleaned")
        self.loader.load_all_stocks()

    def classify_quadrant(self, cagr, volatility):
        if cagr >= 0.12 and volatility < 0.25:
            return "High Return, Low Risk"
        elif cagr >= 0.12 and volatility >= 0.25:
            return "High Return, High Risk"
        elif cagr < 0.12 and volatility < 0.25:
            return "Low Return, Low Risk"
        else:
            return "Low Return, High Risk"

    def generate(self):
        scatter_data = []

        for symbol in self.symbols:
            prices_symbol = f"{symbol}_PRICES"

            try:
                df = self.loader.get_stock(prices_symbol)
            except Exception:
                continue

            if df.empty or len(df) < 252:
                continue

            analytics = StockAnalytics(df)

            cagr = analytics.cagr()
            volatility = analytics.volatility()
            mean_daily_return = analytics.mean_daily_return()

            scatter_data.append({
                "symbol": symbol,
                "return": round(cagr * 100, 2),
                "mean_daily_return": round(mean_daily_return * 100, 4),
                "volatility": round(volatility * 100, 2),  # ✅ FIXED
                "quadrant": self.classify_quadrant(cagr, volatility)
            })

        return scatter_data









# from analytics.stock_metrics import StockAnalytics
# from analytics.stock_loader import StockDataLoader


# class PortfolioReturnRiskScatter:
#     def __init__(self, symbols: list):
#         self.symbols = symbols
#         self.loader = StockDataLoader("data_cleaned")
#         self.loader.load_all_stocks()

#     def classify_quadrant(self, cagr, volatility):
#         if cagr >= 0.12 and volatility < 0.25:
#             return "High Return, Low Risk"
#         elif cagr >= 0.12 and volatility >= 0.25:
#             return "High Return, High Risk"
#         elif cagr < 0.12 and volatility < 0.25:
#             return "Low Return, Low Risk"
#         else:
#             return "Low Return, High Risk"

#     def generate(self):
#         scatter_data = []

#         for symbol in self.symbols:
#             prices_symbol = f"{symbol}_PRICES"

#             try:
#                 df = self.loader.get_stock(prices_symbol)
#             except Exception:
#                 continue

#             if df.empty or len(df) < 252:
#                 continue

#             analytics = StockAnalytics(df)

#             cagr = analytics.cagr()
#             volatility = analytics.volatility()
#             mean_daily_return = analytics.mean_daily_return()  # ✅ FIX

#             scatter_data.append({
#                 "symbol": symbol,
#                 "return": round(cagr * 100, 2),
#                 "mean_daily_return": round(mean_daily_return * 100, 4),
#                 "volatility": round(volatility * 100, 2),
#                 "quadrant": self.classify_quadrant(cagr, volatility)
#             })

#         return scatter_data
