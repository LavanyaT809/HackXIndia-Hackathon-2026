from flask import Blueprint, jsonify
from database.database import SessionLocal
from database.models import Portfolio
from analytics.correlation import PortfolioCorrelation

correlation_bp = Blueprint(
    "correlation",
    __name__,
    url_prefix="/api/portfolio"
)


@correlation_bp.route("/correlation/<user_id>", methods=["GET"])
def portfolio_correlation(user_id):
    db = SessionLocal()

    rows = db.query(Portfolio).filter(
        Portfolio.user_id == user_id
    ).all()

    db.close()

    if not rows:
        return jsonify({"error": "Portfolio is empty"}), 404

    symbols = [
        f"{row.symbol}_PRICES"
        for row in rows if row.weight > 0
    ]

    if len(symbols) < 2:
        return jsonify({
            "error": "At least 2 stocks required for correlation"
        }), 400

    result = PortfolioCorrelation(symbols).summary()

    return jsonify({
        "user_id": user_id,
        **result
    })







# from flask import Blueprint, jsonify
# from database.database import SessionLocal
# from database.models import Portfolio
# from analytics.correlation import PortfolioCorrelation

# correlation_bp = Blueprint(
#     "correlation",
#     __name__,
#     url_prefix="/api/portfolio"
# )


# @correlation_bp.route("/correlation/<user_id>", methods=["GET"])
# def portfolio_correlation(user_id):
#     db = SessionLocal()

#     rows = db.query(Portfolio).filter(
#         Portfolio.user_id == user_id
#     ).all()

#     db.close()

#     if not rows:
#         return jsonify({
#             "error": "Portfolio is empty"
#         }), 404

#     symbols = [
#         f"{row.symbol}_PRICES"
#         for row in rows
#         if row.weight > 0
#     ]

#     if len(symbols) < 2:
#         return jsonify({
#             "error": "At least 2 stocks required for correlation"
#         }), 400

#     correlation = PortfolioCorrelation(symbols)
#     result = correlation.summary()

#     return jsonify({
#         "user_id": user_id,
#         **result
#     })
