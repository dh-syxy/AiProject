import requests

def query_ollama(prompt, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False  # 设置为 True 可以获取流式响应
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"API 请求失败: {response.text}")

# 使用示例
response = query_ollama("你好，请介绍一下你自己")
print(response)