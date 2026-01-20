import os
import pandas as pd


class StockDataLoader:
    """
    Centralized stock data loader for analytics & data science.
    Loads cleaned stock CSVs and provides safe access methods.
    """

    def __init__(self, data_dir="data_cleaned"):
        self.data_dir = data_dir
        self.stocks = {}

    def load_all_stocks(self):
        """
        Load all CSV files from data_cleaned folder.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]

        for file in files:
            symbol = file.replace(".csv", "").upper()
            path = os.path.join(self.data_dir, file)

            try:
                df = pd.read_csv(path)

                # Ensure required columns exist
                required_cols = {
                    "Date", "Close", "High", "Low", "Open",
                    "Volume", "Return", "MA10", "MA50"
                }

                if not required_cols.issubset(df.columns):
                    print(f"⚠️ Skipping {symbol}: Missing required columns")
                    continue

                # Standardize Date
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)

                self.stocks[symbol] = df

            except Exception as e:
                print(f"❌ Failed loading {symbol}: {e}")

        print(f"✅ Loaded {len(self.stocks)} stocks")

    def list_stocks(self):
        """
        Return list of available stock symbols.
        """
        return sorted(self.stocks.keys())

    def get_stock(self, symbol):
        """
        Return DataFrame for a specific stock.
        """
        symbol = symbol.upper()
        if symbol not in self.stocks:
            raise ValueError(f"Stock '{symbol}' not found")
        return self.stocks[symbol].copy()

    def get_date_range(self, symbol, start_date=None, end_date=None):
        """
        Return stock data filtered by date range.
        """
        df = self.get_stock(symbol)

        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        return df.reset_index(drop=True)
