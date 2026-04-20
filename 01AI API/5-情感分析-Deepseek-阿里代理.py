#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dashscope
from dashscope.api_entities.dashscope_response import Role
import os
# 从环境变量中，获取 DASHSCOPE_API_KEY
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 封装模型响应函数
def get_response(messages):
    response = dashscope.Generation.call(
        model='deepseek-r1',  # 使用 deepseek-r1 模型
        messages=messages,
        result_format='message'  # 将输出设置为message形式
    )
    return response

# 测试对话
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "你好，你是什么大模型？"}
]
response = get_response(messages)
print(response.output.choices[0].message.content)

