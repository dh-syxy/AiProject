import requests

# 支持流式响应的query_ollama函数
def query_ollama(prompt, model="deepseek-r1:1.5b", stream=False):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": stream  # 设置为 True 可以获取流式响应
    }
    
    if stream:
        # 流式响应处理
        with requests.post(url, json=data, stream=True) as response:
            if response.status_code == 200:
                # 逐行打印流式响应内容
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # Ollama流式返回每行是一个json字符串
                        try:
                            import json
                            obj = json.loads(line)
                            # 打印每段响应内容
                            print(obj.get("response", ""), end="", flush=True)
                        except Exception as e:
                            print(f"解析流式响应出错: {e}")
            else:
                raise Exception(f"API 请求失败: {response.text}")
    else:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"API 请求失败: {response.text}")

# 使用示例
print("流式响应：")
#query_ollama("你好，请介绍一下你自己", stream=True)
query_ollama("帮我写一个二分查找法", stream=True)