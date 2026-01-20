
from db.database import init_db
from db.portfolio_repo import PortfolioRepository

init_db()
repo = PortfolioRepository()

repo.clear_portfolio()
repo.add_stock("AAAU_PRICES", 0.4)
repo.add_stock("AAA_PRICES", 0.6)

print(repo.get_portfolio())
