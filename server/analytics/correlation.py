import pandas as pd
import numpy as np
from analytics.stock_loader import StockDataLoader


class PortfolioCorrelation:
    def __init__(self, symbols, data_dir="data_cleaned"):
        """
        symbols: ['AAAU_PRICES', 'AAPL_PRICES']
        """
        self.symbols = symbols
        self.loader = StockDataLoader(data_dir)
        self.loader.load_all_stocks()

    def build_returns_df(self):
        returns = []

        for symbol in self.symbols:
            try:
                df = self.loader.get_stock(symbol)
            except Exception:
                continue

            if df.empty or "Return" not in df.columns:
                continue

            temp = df[["Date", "Return"]].copy()
            temp = temp.rename(columns={"Return": symbol})
            temp = temp.set_index("Date")
            returns.append(temp)

        if not returns:
            return pd.DataFrame()

        return pd.concat(returns, axis=1).dropna()

    def correlation_matrix(self):
        returns_df = self.build_returns_df()
        if returns_df.empty:
            return pd.DataFrame()

        return returns_df.corr()

    def diversification_score(self):
        corr = self.correlation_matrix()

        if corr.empty or corr.shape[0] < 2:
            return 0

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        avg_corr = upper.stack().mean()
        return round(1 - avg_corr, 4)

    def summary(self):
        corr = self.correlation_matrix()

        return {
            "diversification_score": self.diversification_score(),
            "correlation_matrix": corr.round(3).to_dict()
        }








# import pandas as pd
# import numpy as np
# from analytics.stock_loader import StockDataLoader


# class PortfolioCorrelation:
#     def __init__(self, symbols, data_dir="data_cleaned"):
#         """
#         symbols: list of stock symbols (e.g. ['AAAU_PRICES', 'AAPL_PRICES'])
#         """
#         self.symbols = symbols
#         self.loader = StockDataLoader(data_dir)
#         self.loader.load_all_stocks()

#     def build_returns_df(self):
#         """
#         Build DataFrame of returns:
#         Date | AAAU | AAPL | MSFT
#         """
#         returns = []

#         for symbol in self.symbols:
#             df = self.loader.get_stock(symbol)

#             if df.empty or "Return" not in df.columns:
#                 continue

#             temp = df[["Date", "Return"]].copy()
#             temp = temp.rename(columns={"Return": symbol})
#             temp = temp.set_index("Date")
#             returns.append(temp)

#         if not returns:
#             return pd.DataFrame()

#         return pd.concat(returns, axis=1).dropna()

#     def correlation_matrix(self):
#         returns_df = self.build_returns_df()
#         if returns_df.empty:
#             return pd.DataFrame()

#         return returns_df.corr()

#     def diversification_score(self):
#         """
#         Diversification Score = 1 - average off-diagonal correlation
#         Range: 0 (bad) → 1 (excellent)
#         """
#         corr = self.correlation_matrix()
#         if corr.empty or len(corr) < 2:
#             return 0

#         upper = corr.where(
#             np.triu(np.ones(corr.shape), k=1).astype(bool)
#         )

#         avg_corr = upper.stack().mean()
#         return round(1 - avg_corr, 4)

#     def summary(self):
#         corr = self.correlation_matrix()

#         return {
#             "diversification_score": self.diversification_score(),
#             "correlation_matrix": corr.round(3).to_dict()
#         }
