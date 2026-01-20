import numpy as np
from analytics.stock_metrics import StockAnalytics

def portfolio_contribution(portfolio_df, stock_data_map):
    """
    portfolio_df:
        symbol | weight
    stock_data_map:
        { "AAPL": df, "MSFT": df }
    """

    results = []
    skipped = []

    total_return = 0.0
    total_risk = 0.0

    for _, row in portfolio_df.iterrows():
        symbol = row["symbol"]
        weight = float(row["weight"])

        # ✅ SAFETY CHECK (THIS FIXES YOUR ERROR)
        if symbol not in stock_data_map:
            print(f"⚠️ Skipping {symbol}: market data not found")
            skipped.append(symbol)
            continue

        analytics = StockAnalytics(stock_data_map[symbol])

        cagr = analytics.cagr()
        vol = analytics.volatility()

        contrib_return = weight * cagr
        contrib_risk = weight * vol

        total_return += contrib_return
        total_risk += contrib_risk

        results.append({
            "symbol": symbol,
            "weight_%": round(weight * 100, 2),
            "return_contribution_%": round(contrib_return * 100, 2),
            "risk_contribution_%": round(contrib_risk * 100, 2)
        })

    return {
        "portfolio_return_%": round(total_return * 100, 2),
        "portfolio_risk_%": round(total_risk * 100, 2),
        "stocks": results,
        "skipped_symbols": skipped
    }








# import numpy as np
# from analytics.stock_metrics import StockAnalytics

# def portfolio_contribution(portfolio_df, stock_data_map):
#     """
#     portfolio_df:
#         symbol | weight
#     stock_data_map:
#         { "AAPL": df, "MSFT": df }
#     """

#     results = []

#     total_return = 0
#     total_risk = 0

#     for _, row in portfolio_df.iterrows():
#         symbol = row["symbol"]
#         weight = row["weight"]

#         analytics = StockAnalytics(stock_data_map[symbol])

#         cagr = analytics.cagr()
#         vol = analytics.volatility()

#         contrib_return = weight * cagr
#         contrib_risk = weight * vol

#         total_return += contrib_return
#         total_risk += contrib_risk

#         results.append({
#             "symbol": symbol,
#             "weight": round(weight * 100, 2),
#             "return_contribution": round(contrib_return * 100, 2),
#             "risk_contribution": round(contrib_risk * 100, 2)
#         })

#     return {
#         "portfolio_return_%": round(total_return * 100, 2),
#         "portfolio_risk_%": round(total_risk * 100, 2),
#         "stocks": results
#     }
