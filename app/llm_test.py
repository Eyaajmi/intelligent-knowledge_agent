import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen2.5:3b-instruct",
    "prompt": "What is RAG?",
    "stream": False
}

response = requests.post(url, json=data)

response.raise_for_status()

result = response.json()

print("Answer:")
print(result["response"])

print("\nPrompt tokens:", result["prompt_eval_count"])
print("Generated tokens:", result["eval_count"])