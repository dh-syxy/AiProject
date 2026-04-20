
# 导入依赖库
import dashscope
import os

# 从环境变量中获取 API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 基于 prompt 生成文本
# 使用 deepseek-v3 模型
def get_completion(prompt, model="deepseek-v3"):
    messages = [{"role": "user", "content": prompt}]    # 将 prompt 作为用户输入
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',  # 将输出设置为message形式
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.output.choices[0].message.content  # 返回模型生成的文本
    
# 任务描述

instruction = """
你的任务是帮用户推荐符合用户需求的理财产品。返回的内容包含三个属性：产品名称，产品收益率，产品期限。
"""

# 用户输入
input_text = """
我想买一个风险较低，但是收益率不错的理财产品。
"""

# prompt 模版。instruction 和 input_text 会被替换为上面的内容
prompt = f"""
# 目标
{instruction}

# 用户输入
{input_text}
"""

print("==== Prompt ====")
print(prompt)
print("================")

# 调用大模型
response = get_completion(prompt)
print(response)
