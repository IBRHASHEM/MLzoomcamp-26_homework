import requests

url = "http://localhost:9696/predict"

customer = {
    "geography": "France",
    "gender": "Female",
    "hascrcard": 1,
    "isactivemember": 1,
    "complain": 0,
    "satisfaction_score": 3,
    "card_type": "DIAMOND",
    "creditscore": 650.0,
    "age": 42,
    "tenure": 3,
    "balance": 75000.0,
    "numofproducts": 1,
    "estimatedsalary": 85000.0,
    "point_earned": 450
}

try:
    response = requests.post(url, json=customer)
    response.raise_for_status()
    result = response.json()
    print("API Response:")
    print(result)
    if result.get("will_churn"):
        print(f"Warning: Customer is likely to churn! Probability: {result['churn_probability']:.4f}")
    else:
        print(f"Customer is likely to stay. Probability: {result['churn_probability']:.4f}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Make sure the server is running on http://localhost:9696")
except Exception as e:
    print(f"An error occurred: {e}")
