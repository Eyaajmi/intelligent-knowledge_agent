import requests

url = "http://localhost:11434/api/chat"

data = {
    "model": "qwen2.5:3b-instruct",
    "messages": [
        {
            "role": "system",
            "content": "You are a patient AI teacher. Explain technical concepts to complete beginners using simple examples."
        },
        {
            "role": "user",
            "content": "What is RAG?"
        }
    ],
    "stream": False
}

response = requests.post(url, json=data)

response.raise_for_status()

result = response.json()

print(result["message"]["content"])