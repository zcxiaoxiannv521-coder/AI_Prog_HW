from src.llm_qa import llm_answer
# 1. 时间需求查询
def query_time(df, question):
    # 原因分析问题交给大模型
    reason_words = [
        "为什么",
        "原因",
        "影响因素",
        "分析"
    ]
    if any(word in question for word in reason_words):
        return None
    # 高峰订单数量
    if ("高峰" in question
        and any(word in question for word in ["多少", "数量", "多少条", "几单"])):

        count = df[df["is_peak_hour"] == 1].shape[0]

        return (
            f"高峰时段订单数量约为 {count} 条。"
        )
    # 夜间订单数量
    if (
        ("夜间" in question or "晚上" in question)
        and any(word in question for word in ["多少", "数量", "多少条", "几单"])
    ):
        count = df[df["is_night"] == 1].shape[0]
        return (
            f"夜间订单数量约为 {count} 条。"
        )
    return None
# 2. 区域排名查询
def query_region(df, question):
    reason_words = [
        "为什么",
        "原因",
        "影响因素",
        "分析"
    ]
    if any(word in question for word in reason_words):
        return None
    if (
        ("区域" in question or "地点" in question)
        and any(word in question for word in [
            "最多",
            "最高",
            "排名",
            "top",
            "哪个"
        ])
    ):
        top_region = (
            df["PULocationID"]
            .value_counts()
            .head(5)
        )
        return (
            "出租车订单量最高的TOP5区域：\n"
            f"{top_region}\n\n"
            "相关分析图表：outputs/m2_2_region.png"
        )
    return None
# 3. 费用查询
def query_fare(df, question):
    if (
        ("车费" in question or "费用" in question)
        and not ("为什么" in question)
    ):
        mean_fare = df["fare_amount"].mean()
        return (
            f"出租车平均车费约为 {mean_fare:.2f} 美元。\n"
            "相关分析图表：outputs/m2_3_fare.png"
        )
    return None
# 4. 模型评价查询
def query_model(question):
    if "模型" in question or "预测" in question:
        return """
模型性能分析：

神经网络：

1. 可以学习复杂非线性关系；
2. 对大量数据具有较强拟合能力；
3. 适合出租车需求预测任务。


随机森林：

1. 训练速度较快；
2. 对异常值具有较强鲁棒性；
3. 模型解释能力较强。


本任务中：

神经网络适合学习复杂时空需求变化规律；

随机森林作为传统机器学习方法，
用于模型性能对比。


相关文件：
outputs/m3_model_metrics.csv
"""
    return None
# 5. 数据统计查询
def query_statistics(df, question):
    if (
        "多少条" in question
        or "数据量" in question
        or "记录数" in question
    ):
        return (
            f"当前出租车数据集共有 {len(df)} 条记录。\n"
            "数据来源：NYC Yellow Taxi Trip Records"
        )
    if "平均距离" in question:
        mean_distance = df["trip_distance"].mean()
        return (
            f"平均行程距离约为 {mean_distance:.2f} mile。"
        )
    return None
# 主问答系统
def qa_system(df):
    print("=" * 30)
    print("出租车智能问答系统")
    print("=" * 30)
    print("""
支持的问题类型：

1. 时段需求查询
2. 区域排名查询
3. 费用查询
4. 模型评价查询
5. 数据统计查询


如果问题无法通过规则回答，
系统将调用大模型解释。


输入“退出”结束系统。
""")
    while True:
        question = input("\n请输入问题：")
        if question == "退出":
            print("系统结束")
            break
        answer = None
        # 规则匹配顺序
        rule_functions = [
            query_time,
            query_region,
            query_fare,
            query_statistics

        ]
        for func in rule_functions:
            answer = func(df, question)
            if answer:
                break
        # 模型查询
        if answer is None:
            answer = query_model(question)
        # 大模型兜底
        if answer is None:
            print("\n规则无法匹配，调用大模型...")
            answer = llm_answer(
                question,
                """
你是一个纽约出租车数据分析助手。


已知背景：

1. 数据来源为 NYC Yellow Taxi Trip Records；
2. 已完成订单数量、时间规律、
区域热度、费用关系等分析；
3. 如果用户询问原因、趋势、影响因素，
需要结合城市交通规律进行解释。


回答要求：

1. 不编造具体数据；
2. 给出合理交通原因；
3. 语言简洁。


"""
            )
        print("\n系统回答：")
        print(answer)