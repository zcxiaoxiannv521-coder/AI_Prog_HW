import gradio as gr
from src.m1_data_processing import data_processing
from src.m4_qa_system import (
    query_time,
    query_region,
    query_fare,
    query_model,
    query_statistics
)

# 加载数据
df = data_processing()
def chatbot(question):
    answer = None
    functions = [
        query_time,
        query_region,
        query_fare,
        query_statistics
    ]
    for func in functions:
        answer = func(df, question)
        if answer:
            break
    if answer is None:
        answer = query_model(question)
    if answer is None:
        answer = (
            "无法识别该问题，请尝试：\n"
            "1. 查询高峰订单量\n"
            "2. 哪个区域订单最多\n"
            "3. 平均车费是多少\n"
            "4. 模型哪个好"
        )
    return answer

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="请输入出租车相关问题",placeholder="例如：哪个区域订单最多？"),
    outputs=gr.Textbox(label="智能回答"),title="NYC出租车智能分析助手",
    description=
    """
基于纽约出租车公开数据集，
实现数据统计、区域分析、
需求查询以及模型分析。
"""
)
if __name__ == "__main__":
    demo.launch()