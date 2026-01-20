from flask import Blueprint, jsonify
from database.database import SessionLocal
from database.models import Portfolio

from analytics.portfolio_scatter import PortfolioReturnRiskScatter
from analytics.stock_loader import StockDataLoader
from analytics.stock_metrics import StockAnalytics
from analytics.portfolio_contribution import portfolio_contribution
from analytics.portfolio_analytics import PortfolioAnalytics
from analytics.portfolio_insights import generate_portfolio_insights
from analytics.correlation import PortfolioCorrelation

import pandas as pd

portfolio_analysis_bp = Blueprint(
    "portfolio_analysis",
    __name__,
    url_prefix="/api/portfolio"
)


@portfolio_analysis_bp.route("/analysis/<user_id>", methods=["GET"])
def analyze_portfolio(user_id):
    db = SessionLocal()
    rows = db.query(Portfolio).filter(
        Portfolio.user_id == user_id
    ).all()
    db.close()

    if not rows:
        return jsonify({"error": "Portfolio is empty"}), 404

    portfolio_df = pd.DataFrame([
        {"symbol": r.symbol.upper(), "weight": r.weight}
        for r in rows if r.weight > 0
    ])

    portfolio_df["weight"] /= portfolio_df["weight"].sum()

    loader = StockDataLoader("data_cleaned")
    loader.load_all_stocks()

    # ---------- Stock Metrics ----------
    stock_metrics = {}
    for symbol in portfolio_df["symbol"]:
        try:
            df = loader.get_stock(f"{symbol}_PRICES")
            stock_metrics[symbol] = StockAnalytics(df).summary()
        except Exception:
            continue

    # ---------- Contribution ----------
    contribution = portfolio_contribution(
        portfolio_df,
        {s: loader.get_stock(f"{s}_PRICES") for s in portfolio_df["symbol"]}
    )

    # ---------- Portfolio Summary ----------
    portfolio_weights = {
        f"{row['symbol']}_PRICES": row["weight"]
        for _, row in portfolio_df.iterrows()
    }

    portfolio_analytics = PortfolioAnalytics(loader, portfolio_weights)
    portfolio_summary = portfolio_analytics.summary()

    # ---------- Scatter ----------
    scatter_data = PortfolioReturnRiskScatter(
        list(portfolio_df["symbol"])
    ).generate()

    # ---------- Correlation ----------
    symbols = [f"{s}_PRICES" for s in portfolio_df["symbol"]]
    correlation_summary = PortfolioCorrelation(symbols).summary()

    # ---------- Insights ----------
    insights = generate_portfolio_insights(
        scatter_data,
        portfolio_summary["risk"]
    )

    return jsonify({
        "user_id": user_id,
        "portfolio_summary": portfolio_summary,
        "stocks": stock_metrics,
        "contribution": contribution,
        "scatter": scatter_data,
        "correlation_matrix": correlation_summary["correlation_matrix"],
        "diversification_score": correlation_summary["diversification_score"],
        "insights": insights
    })









# from flask import Blueprint, jsonify
# from database.database import SessionLocal
# from database.models import Portfolio

# from analytics.portfolio_scatter import PortfolioReturnRiskScatter
# from analytics.stock_loader import StockDataLoader
# from analytics.stock_metrics import StockAnalytics
# from analytics.portfolio_contribution import portfolio_contribution
# from analytics.portfolio_analytics import PortfolioAnalytics
# from analytics.portfolio_insights import generate_portfolio_insights
# from analytics.correlation import PortfolioCorrelation


# import pandas as pd

# portfolio_analysis_bp = Blueprint(
#     "portfolio_analysis",
#     __name__,
#     url_prefix="/api/portfolio"
# )

# # ==========================================================
# # 📊 FULL PORTFOLIO ANALYSIS (USED BY FRONTEND MAIN PAGE)
# # ==========================================================
# @portfolio_analysis_bp.route("/analysis/<user_id>", methods=["GET"])
# def analyze_portfolio(user_id):
#     db = SessionLocal()
#     rows = db.query(Portfolio).filter(
#         Portfolio.user_id == user_id
#     ).all()
#     db.close()

#     if not rows:
#         return jsonify({"error": "Portfolio is empty"}), 404

#     # ----------------- Build Portfolio DF -----------------
#     portfolio_df = pd.DataFrame([
#         {"symbol": r.symbol.upper(), "weight": r.weight}
#         for r in rows if r.weight > 0
#     ])

#     portfolio_df["weight"] /= portfolio_df["weight"].sum()

#     # ----------------- Load Stock Data -----------------
#     loader = StockDataLoader("data_cleaned")
#     loader.load_all_stocks()

#     # ----------------- Individual Stock Metrics -----------------
#     stock_metrics = {}
#     for symbol in portfolio_df["symbol"]:
#         try:
#             df = loader.get_stock(f"{symbol}_PRICES")
#             stock_metrics[symbol] = StockAnalytics(df).summary()
#         except Exception:
#             continue

#     # ----------------- Contribution -----------------
#     contribution = portfolio_contribution(
#         portfolio_df,
#         {s: loader.get_stock(f"{s}_PRICES") for s in portfolio_df["symbol"]}
#     )

#     # ----------------- Portfolio Summary -----------------
#     portfolio_weights = {
#         f"{row['symbol']}_PRICES": row["weight"]
#         for _, row in portfolio_df.iterrows()
#     }

#     portfolio_analytics = PortfolioAnalytics(loader, portfolio_weights)
#     portfolio_summary = portfolio_analytics.summary()

#     # ----------------- Scatter (Risk–Return) -----------------
#     scatter_data = PortfolioReturnRiskScatter(
#         list(portfolio_df["symbol"])
#     ).generate()

#     # ----------------- Correlation & Diversification -----------------
#     correlation_obj = PortfolioCorrelation(
#         loader,
#         list(portfolio_df["symbol"])
#     )
#     correlation_summary = correlation_obj.summary()

#     # ----------------- Insights -----------------
#     insights = generate_portfolio_insights(
#         scatter_data,
#         portfolio_summary["risk"]
#     )

#     # ----------------- FINAL RESPONSE -----------------
#     return jsonify({
#         "user_id": user_id,
#         "portfolio_summary": portfolio_summary,
#         "stocks": stock_metrics,
#         "contribution": contribution,
#         "scatter": scatter_data,
#         "correlation_matrix": correlation_summary["correlation_matrix"],  # ✅ heatmap
#         "diversification_score": correlation_summary["diversification_score"],
#         "insights": insights
#     })


# # ==========================================================
# # 📈 SCATTER ONLY (USED BY SCATTER API)
# # ==========================================================
# @portfolio_analysis_bp.route("/scatter/<user_id>", methods=["GET"])
# def portfolio_scatter(user_id):
#     db = SessionLocal()
#     rows = db.query(Portfolio).filter(
#         Portfolio.user_id == user_id
#     ).all()
#     db.close()

#     symbols = list({r.symbol.upper() for r in rows})

#     return jsonify(
#         PortfolioReturnRiskScatter(symbols).generate()
#     )








# from flask import Blueprint, jsonify
# from database.database import SessionLocal
# from database.models import Portfolio

# from analytics.portfolio_scatter import PortfolioReturnRiskScatter
# from analytics.stock_loader import StockDataLoader
# from analytics.stock_metrics import StockAnalytics
# from analytics.portfolio_contribution import portfolio_contribution
# from analytics.portfolio_analytics import PortfolioAnalytics
# from analytics.portfolio_insights import generate_portfolio_insights

# import pandas as pd

# portfolio_analysis_bp = Blueprint(
#     "portfolio_analysis",
#     __name__,
#     url_prefix="/api/portfolio"
# )

# @portfolio_analysis_bp.route("/analysis/<user_id>", methods=["GET"])
# def analyze_portfolio(user_id):
#     db = SessionLocal()
#     rows = db.query(Portfolio).filter(
#         Portfolio.user_id == user_id
#     ).all()
#     db.close()

#     if not rows:
#         return jsonify({"error": "Portfolio is empty"}), 404

#     portfolio_df = pd.DataFrame([
#         {"symbol": r.symbol.upper(), "weight": r.weight}
#         for r in rows if r.weight > 0
#     ])

#     portfolio_df["weight"] /= portfolio_df["weight"].sum()

#     loader = StockDataLoader("data_cleaned")
#     loader.load_all_stocks()

#     stock_metrics = {}
#     for symbol in portfolio_df["symbol"]:
#         try:
#             df = loader.get_stock(f"{symbol}_PRICES")
#             stock_metrics[symbol] = StockAnalytics(df).summary()
#         except Exception:
#             continue

#     contribution = portfolio_contribution(
#         portfolio_df,
#         {s: loader.get_stock(f"{s}_PRICES") for s in portfolio_df["symbol"]}
#     )

#     portfolio_weights = {
#         f"{row['symbol']}_PRICES": row["weight"]
#         for _, row in portfolio_df.iterrows()
#     }

#     portfolio_analytics = PortfolioAnalytics(loader, portfolio_weights)
#     portfolio_summary = portfolio_analytics.summary()

#     scatter_data = PortfolioReturnRiskScatter(
#         list(portfolio_df["symbol"])
#     ).generate()

#     insights = generate_portfolio_insights(
#         scatter_data,
#         portfolio_summary["risk"]
#     )

#     return jsonify({
#         "user_id": user_id,
#         "portfolio_summary": portfolio_summary,
#         "stocks": stock_metrics,
#         "contribution": contribution,
#         "scatter": scatter_data,
#         "insights": insights
#     })


# @portfolio_analysis_bp.route("/scatter/<user_id>", methods=["GET"])
# def portfolio_scatter(user_id):
#     db = SessionLocal()
#     rows = db.query(Portfolio).filter(
#         Portfolio.user_id == user_id
#     ).all()
#     db.close()

#     symbols = list({r.symbol.upper() for r in rows})
#     return jsonify(
#         PortfolioReturnRiskScatter(symbols).generate()
#     )




