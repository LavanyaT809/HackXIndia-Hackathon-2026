from flask import Blueprint, request, jsonify
from database.database import SessionLocal
from database.models import Portfolio

portfolio_bp = Blueprint("portfolio", __name__, url_prefix="/api/portfolio")


# 1️⃣ ADD STOCK TO PORTFOLIO
@portfolio_bp.route("/add", methods=["POST"])
def add_to_portfolio():
    db = SessionLocal()
    data = request.json

    user_id = data.get("user_id")
    symbol = data.get("symbol")

    quantity = data.get("quantity", 0.0)
    buy_price = data.get("buy_price", 0.0)
    weight = data.get("weight", 0.0)

    if not user_id or not symbol:
        return jsonify({
            "error": "user_id and symbol are required"
        }), 400

    entry = Portfolio(
        user_id=user_id,
        symbol=symbol.upper(),
        quantity=float(quantity),
        buy_price=float(buy_price),
        weight=float(weight)
    )

    db.add(entry)
    db.commit()
    db.refresh(entry)
    db.close()

    return jsonify({
        "message": "Stock added to portfolio",
        "id": entry.id,
        "symbol": entry.symbol
    }), 201


# 2️⃣ FETCH USER PORTFOLIO
@portfolio_bp.route("/<user_id>", methods=["GET"])
def get_portfolio(user_id):
    db = SessionLocal()

    rows = db.query(Portfolio).filter(
        Portfolio.user_id == user_id
    ).all()

    db.close()

    return jsonify([
        {
            "id": r.id,
            "symbol": r.symbol,
            "quantity": r.quantity,
            "buy_price": r.buy_price,
            "weight": r.weight,
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ])


# 3️⃣ DELETE STOCK FROM PORTFOLIO
@portfolio_bp.route("/delete/<int:stock_id>", methods=["DELETE"])
def delete_stock(stock_id):
    db = SessionLocal()

    stock = db.query(Portfolio).filter(
        Portfolio.id == stock_id
    ).first()

    if not stock:
        db.close()
        return jsonify({"error": "Stock not found"}), 404

    db.delete(stock)
    db.commit()
    db.close()

    return jsonify({
        "message": "Stock deleted successfully"
    })






# from flask import Blueprint, request, jsonify
# from database.database import SessionLocal
# from database.models import Portfolio

# portfolio_bp = Blueprint("portfolio", __name__, url_prefix="/api/portfolio")


# @portfolio_bp.route("/add", methods=["POST"])
# def add_to_portfolio():
#     db = SessionLocal()
#     data = request.json

#     # required
#     user_id = data.get("user_id")
#     symbol = data.get("symbol")

#     # optional (safe defaults)
#     quantity = data.get("quantity", 0.0)
#     buy_price = data.get("buy_price", 0.0)
#     weight = data.get("weight", 0.0)

#     if not user_id or not symbol:
#         return jsonify({
#             "error": "user_id and symbol are required"
#         }), 400

#     entry = Portfolio(
#         user_id=user_id,
#         symbol=symbol.upper(),
#         quantity=float(quantity),
#         buy_price=float(buy_price),
#         weight=float(weight)
#     )

#     db.add(entry)
#     db.commit()
#     db.refresh(entry)
#     db.close()

#     return jsonify({
#         "message": "Stock added to portfolio",
#         "id": entry.id,
#         "symbol": entry.symbol
#     }), 201
