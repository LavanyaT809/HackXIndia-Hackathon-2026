
# server/analytics/return_risk_scatter.py

import numpy as np
from analytics.stock_loader import StockDataLoader
from analytics.stock_metrics import StockAnalytics


class ReturnRiskScatter:
    def __init__(self):
        self.loader = StockDataLoader()
        self.loader.load_all_stocks()

    def classify_quadrant(self, cagr, volatility):
        """
        Human readable quadrant classification
        """
        if cagr >= 0.12 and volatility < 0.25:
            return "High Return, Low Risk"
        elif cagr >= 0.12 and volatility >= 0.25:
            return "High Return, High Risk"
        elif cagr < 0.12 and volatility < 0.25:
            return "Low Return, Low Risk"
        else:
            return "Low Return, High Risk"

    def generate(self):
        """
        Returns list of dicts for scatter plot
        """
        scatter_data = []

        for symbol in self.loader.list_stocks():
            df = self.loader.get_stock(symbol)

            if df.empty or len(df) < 252:
                continue

            analytics = StockAnalytics(df)

            cagr = analytics.cagr()
            volatility = analytics.volatility()

            scatter_data.append({
                "symbol": symbol.replace("_PRICES", ""),
                "return": round(cagr * 100, 2),
                "volatility": round(volatility * 100, 2),
                "quadrant": self.classify_quadrant(cagr, volatility)
            })

        return scatter_data
