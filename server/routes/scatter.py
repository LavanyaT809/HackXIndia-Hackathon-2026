
from flask import Blueprint, jsonify
from analytics.return_risk_scatter import ReturnRiskScatter

scatter_bp = Blueprint(
    "scatter",
    __name__,
    url_prefix="/api/analytics"
)

@scatter_bp.route("/risk-return-scatter", methods=["GET"])
def risk_return_scatter():
    scatter = ReturnRiskScatter()
    data = scatter.generate()
    return jsonify(data)
