#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
# 从环境变量中，获取 DASHSCOPE_API_KEY
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 封装模型响应函数
def get_response(messages):
    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )
    return response

content = [
    {'image': 'https://aiwucai.oss-cn-huhehaote.aliyuncs.com/pdf_table.jpg'}, # Either a local path or an url
    {'text': '这是一个表格图片，帮我提取里面的内容，输出JSON格式'}
]

messages=[{"role": "user", "content": content}]
# 得到响应
response = get_response(messages)
response


# In[2]:


print(response.output.choices[0].message.content[0]['text'])

