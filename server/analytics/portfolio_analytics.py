import numpy as np
import pandas as pd


class PortfolioAnalytics:
    def __init__(self, loader, portfolio: dict):
        self.loader = loader
        self.portfolio = portfolio
        self.returns_df = self._build_returns_df()

    def _build_returns_df(self):
        dfs = []

        for symbol, weight in self.portfolio.items():
            try:
                df = self.loader.get_stock(symbol)
            except Exception:
                continue

            if df.empty or "Return" not in df.columns:
                continue

            temp = df[["Date", "Return"]].copy()
            temp["weighted_return"] = temp["Return"] * weight
            temp = temp.set_index("Date")
            dfs.append(temp[["weighted_return"]])

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1).fillna(0)

    def portfolio_returns(self):
        if self.returns_df.empty:
            return pd.Series(dtype=float)
        return self.returns_df.sum(axis=1)

    def mean_daily_return(self):
        ret = self.portfolio_returns()
        return float(ret.mean()) if not ret.empty else 0

    def portfolio_volatility(self):
        ret = self.portfolio_returns()
        return float(ret.std() * np.sqrt(252)) if not ret.empty else 0

    def portfolio_cagr(self):
        ret = self.portfolio_returns()
        if ret.empty:
            return 0

        cumulative = (1 + ret).cumprod()
        years = len(ret) / 252
        return float(cumulative.iloc[-1] ** (1 / years) - 1)

    def max_drawdown(self):
        ret = self.portfolio_returns()
        if ret.empty:
            return 0

        cumulative = (1 + ret).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    def risk_label(self, vol):
        if vol < 0.10:
            return "Low Risk"
        elif vol < 0.20:
            return "Medium Risk"
        return "High Risk"

    def summary(self):
        vol = self.portfolio_volatility()
        return {
            "portfolio_cagr": round(self.portfolio_cagr(), 4),
            "portfolio_volatility": round(vol, 4),
            "portfolio_max_drawdown": round(self.max_drawdown(), 4),
            "mean_daily_return": round(self.mean_daily_return(), 6),
            "risk": self.risk_label(vol)
        }







# import numpy as np
# import pandas as pd

# class PortfolioAnalytics:
#     def __init__(self, loader, portfolio: dict):
#         self.loader = loader
#         self.portfolio = portfolio
#         self.returns_df = self._build_returns_df()

#     def _build_returns_df(self):
#         dfs = []

#         for symbol, weight in self.portfolio.items():
#             try:
#                 df = self.loader.get_stock(symbol)
#             except Exception:
#                 continue

#             if df.empty or "Return" not in df.columns:
#                 continue

#             temp = df[["Date", "Return"]].copy()
#             temp["weighted_return"] = temp["Return"] * weight
#             temp = temp.set_index("Date")
#             dfs.append(temp[["weighted_return"]])

#         if not dfs:
#             return pd.DataFrame()

#         return pd.concat(dfs, axis=1).fillna(0)

#     def portfolio_returns(self):
#         return self.returns_df.sum(axis=1) if not self.returns_df.empty else pd.Series(dtype=float)

#     def mean_daily_return(self):
#         ret = self.portfolio_returns()
#         return float(ret.mean()) if not ret.empty else 0

#     def portfolio_volatility(self):
#         ret = self.portfolio_returns()
#         return float(ret.std() * np.sqrt(252)) if not ret.empty else 0

#     def portfolio_cagr(self):
#         ret = self.portfolio_returns()
#         if ret.empty:
#             return 0
#         cumulative = (1 + ret).cumprod()
#         years = len(ret) / 252
#         return float(cumulative.iloc[-1] ** (1 / years) - 1)

#     def max_drawdown(self):
#         ret = self.portfolio_returns()
#         if ret.empty:
#             return 0
#         cumulative = (1 + ret).cumprod()
#         peak = cumulative.cummax()
#         drawdown = (cumulative - peak) / peak
#         return float(drawdown.min())

#     def risk_label(self, vol):
#         if vol < 0.10:
#             return "Low Risk"
#         elif vol < 0.20:
#             return "Medium Risk"
#         return "High Risk"

#     def summary(self):
#         vol = self.portfolio_volatility()
#         return {
#             "portfolio_cagr": round(self.portfolio_cagr(), 4),
#             "portfolio_volatility": round(vol, 4),
#             "portfolio_max_drawdown": round(self.max_drawdown(), 4),
#             "mean_daily_return": round(self.mean_daily_return(), 6),
#             "risk": self.risk_label(vol)
#         }





# import numpy as np
# import pandas as pd

# class PortfolioAnalytics:
#     def __init__(self, loader, portfolio: dict):
#         self.loader = loader
#         self.portfolio = portfolio
#         self.returns_df = self._build_returns_df()

#     def _build_returns_df(self):
#         dfs = []

#         for symbol, weight in self.portfolio.items():
#             try:
#                 df = self.loader.get_stock(symbol)
#             except Exception:
#                 continue

#             if df.empty or "Return" not in df.columns:
#                 continue

#             temp = df[["Date", "Return"]].copy()
#             temp["weighted_return"] = temp["Return"] * weight
#             temp = temp.set_index("Date")
#             dfs.append(temp[["weighted_return"]])

#         if not dfs:
#             return pd.DataFrame()

#         return pd.concat(dfs, axis=1).fillna(0)

#     def portfolio_returns(self):
#         if self.returns_df.empty:
#             return pd.Series(dtype=float)
#         return self.returns_df.sum(axis=1)

#     def mean_daily_return(self):
#         ret = self.portfolio_returns()
#         return float(ret.mean()) if not ret.empty else 0

#     def portfolio_volatility(self):
#         ret = self.portfolio_returns()
#         return float(ret.std() * np.sqrt(252)) if not ret.empty else 0

#     def portfolio_cagr(self):
#         ret = self.portfolio_returns()
#         if ret.empty:
#             return 0

#         cumulative = (1 + ret).cumprod()
#         years = len(ret) / 252
#         return float(cumulative.iloc[-1] ** (1 / years) - 1)

#     def max_drawdown(self):
#         ret = self.portfolio_returns()
#         if ret.empty:
#             return 0

#         cumulative = (1 + ret).cumprod()
#         peak = cumulative.cummax()
#         drawdown = (cumulative - peak) / peak
#         return float(drawdown.min())

#     def risk_label(self, volatility):
#         if volatility < 0.10:
#             return "Low Risk"
#         elif volatility < 0.20:
#             return "Medium Risk"
#         return "High Risk"

#     def summary(self):
#         vol = self.portfolio_volatility()
#         return {
#             "portfolio_cagr": round(self.portfolio_cagr(), 4),          # decimal
#             "portfolio_volatility": round(vol, 4),                     # decimal
#             "portfolio_max_drawdown": round(self.max_drawdown(), 4),   # decimal
#             "mean_daily_return": round(self.mean_daily_return(), 6),
#             "risk": self.risk_label(vol)
#         }

