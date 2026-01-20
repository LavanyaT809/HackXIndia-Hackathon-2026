
def generate_portfolio_insights(scatter_data, risk_label):
    insights = []

    # High-performing assets
    for s in scatter_data:
        if s["quadrant"] == "High Return, Low Risk":
            insights.append(
                f"{s['symbol']} offers strong returns with comparatively low risk."
            )

    # Portfolio-level insight
    insights.append(
        f"Overall portfolio risk is classified as {risk_label}, based on volatility."
    )

    return insights
