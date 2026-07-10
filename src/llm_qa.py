import os
from openai import OpenAI
client = OpenAI(
    api_key="sk-a389f37205b2464fbaafc4d3a4928567",
    base_url="https://api.deepseek.com"
)
SYSTEM_PROMPT = """
你是一个纽约出租车数据分析助手。

你的任务：
1. 判断用户问题属于：
时间需求分析、区域分析、费用分析、模型分析、数据统计。

2. 根据Python统计结果回答。

3. 不允许编造数据。

4. 回答包含：
数据结论；
原因解释；
相关文件路径。
"""
def llm_answer(question, data_result):

    response = client.chat.completions.create(
        model="deepseek-chat",

        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },

            {
                "role":"user",
                "content":
                f"""
用户问题：
{question}

已有分析结果：
{data_result}

请生成解释性回答。
"""
            }
        ],

        temperature=0.3
    )
    return response.choices[0].message.content