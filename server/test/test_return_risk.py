
from analytics.return_risk_scatter import ReturnRiskScatter

rr = ReturnRiskScatter()
data = rr.generate()

print("Total stocks analyzed:", len(data))
print("\nSample output:\n")

for item in data[:5]:
    print(item)
