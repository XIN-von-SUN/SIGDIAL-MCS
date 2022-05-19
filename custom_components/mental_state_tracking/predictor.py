import requests

API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({"inputs": ["I was not good this morning because I was late. But I passed the final exam and got a great grade.",
                            "so cool!",
                            "not good, becasue it rains all day!"]})

print(f'output is:\n{output}')