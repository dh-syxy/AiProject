import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    json={"prompt": "你好，请介绍一下你自己"}
)
print(response.json())