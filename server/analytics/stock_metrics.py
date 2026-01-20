# server/analytics/stock_metrics.py

import numpy as np
import pandas as pd

class StockAnalytics:
    def __init__(self, df: pd.DataFrame):
        """
        df must contain:
        Date, Close, Volume, Return, MA10, MA50
        """
        self.df = df.copy()

        # Ensure Date is datetime
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df.sort_values("Date", inplace=True)

    # ---------------- BASIC METRICS ----------------
    def mean_daily_return(self):
        return self.df["Return"].mean()

    def annualized_return(self):
        return self.mean_daily_return() * 252

    def volatility(self):
        return self.df["Return"].std() * np.sqrt(252)

    # ---------------- GROWTH METRICS ----------------
    def cagr(self):
        start_price = self.df["Close"].iloc[0]
        end_price = self.df["Close"].iloc[-1]

        days = (self.df["Date"].iloc[-1] - self.df["Date"].iloc[0]).days
        years = days / 365.25

        if years <= 0:
            return 0

        return (end_price / start_price) ** (1 / years) - 1

    # ---------------- DRAWDOWN ----------------
    def max_drawdown(self):
        cumulative = (1 + self.df["Return"]).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    # ---------------- VOLUME ----------------
    def avg_volume(self):
        return self.df["Volume"].mean()

    def volume_trend(self):
        first_30 = self.df["Volume"].iloc[:30].mean()
        last_30 = self.df["Volume"].iloc[-30:].mean()

        if first_30 == 0:
            return 0

        return last_30 / first_30

    # ---------------- RISK SCORE ----------------
    def risk_score(self):
        vol = self.volatility()

        if vol < 0.2:
            return "Low"
        elif vol < 0.35:
            return "Medium"
        else:
            return "High"

    # ---------------- EXPLAINABLE RISK LABEL ----------------
    def risk_label_explained(self):
        vol = self.volatility()
        drawdown = self.max_drawdown()
        cagr = self.cagr()

        if vol < 0.15 and drawdown > -0.20:
            return {
                "label": "Low Risk",
                "explanation": "Low volatility and limited drawdowns indicate stable price behavior"
            }

        elif vol < 0.30 and drawdown > -0.35:
            return {
                "label": "Medium Risk",
                "explanation": "Moderate volatility with acceptable historical losses"
            }

        else:
            return {
                "label": "High Risk",
                "explanation": "High volatility or deep drawdowns indicate higher uncertainty"
            }

    # ---------------- SUMMARY ----------------
    def summary(self):
        risk_info = self.risk_label_explained()

        return {
            "mean_daily_return": round(self.mean_daily_return(), 6),
            "annualized_return": round(self.annualized_return(), 4),
            "volatility": round(self.volatility(), 4),
            "cagr": round(self.cagr(), 4),
            "max_drawdown": round(self.max_drawdown(), 4),
            "avg_volume": int(self.avg_volume()),
            "volume_trend": round(self.volume_trend(), 2),
            "risk": self.risk_score(),
            "risk_label": risk_info["label"],
            "risk_explanation": risk_info["explanation"]
        }
