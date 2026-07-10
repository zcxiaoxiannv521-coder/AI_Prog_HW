import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os


DATA_PATH = "data/yellow_tripdata_2026-01.parquet"
OUTPUT_PATH = "outputs"



def data_processing():
    # 使用 Apache Arrow 读取 Parquet
    table = pq.read_table(DATA_PATH)
    # 转换为 pandas DataFrame 便于分析
    df = table.to_pandas()

    # 数据质量报告（缺失率 + 异常值）
    report = []
    # 缺失率
    missing_rate = df.isnull().mean()
    for col in df.columns:
        report.append({
            "column": col,
            "missing_rate": missing_rate[col]
        })
    report_df = pd.DataFrame(report)
    # 异常值统计（IQR方法）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_list = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

        outlier_list.append({
            "column": col,
            "outlier_count": outlier_count
        })

    outlier_df = pd.DataFrame(outlier_list)
    # 合并数据质量报告
    quality_report = report_df.merge(outlier_df, on="column", how="left")
    # 保存报告
    quality_report.to_csv("outputs/data_quality_report.csv", index=False)

    # 数据清洗策略
    df_clean = df.copy()

    # 去重
    # 原因：
    # 1.避免重复记录对统计分析造成偏差
    df_clean = df_clean.drop_duplicates()

    # 缺失值处理
    for col in df_clean.columns:
        if df_clean[col].dtype in ["float64", "int64"]:
            # 数值型变量缺失值处理：（使用“中位数”填充，而不是均值）
            # 原因：
            # 1. 出租车数据中存在极端值（如超长行程、高额费用）
            # 2. 均值容易被异常值拉偏
            # 3. 中位数对异常值不敏感，更能代表数据中心趋势
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # 类别型变量缺失值处理：（使用“众数”填充（出现频率最高的值））
            # 原因：
            # 1. 类别变量没有大小关系，不能用均值
            # 2. 众数代表最常见状态，符合真实数据分布
            # 3. 能最大程度保持原有分布结构
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 异常值处理（IQR截断）此方法对极端值不敏感，不要求数据服从正态分布，适用于出租车真实业务数据中的异常检测。
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # 截断异常值（保留数据结构，避免删除样本）
        # 原因：
        # 1.极端行程和高额车费可能是真实业务记录，直接删除会造成信息损失。
        # 2.截断能够降低异常值对分析和建模的影响，同时保留样本规模和整体数据结构。
        df_clean[col] = df_clean[col].clip(lower, upper)

    # 时间特征工程

    # NYC Taxi 常见字段
    pickup_col = "tpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime"

    df_clean[pickup_col] = pd.to_datetime(df_clean[pickup_col])
    df_clean[dropoff_col] = pd.to_datetime(df_clean[dropoff_col])

    # 小时
    df_clean["pickup_hour"] = df_clean[pickup_col].dt.hour
    # 星期
    df_clean["pickup_weekday"] = df_clean[pickup_col].dt.weekday
    # 是否周末
    df_clean["is_weekend"] = df_clean["pickup_weekday"].isin([5, 6]).astype(int)
    # 是否高峰（早7-10 / 晚17-20）
    df_clean["is_peak_hour"] = df_clean["pickup_hour"].isin(
        [7, 8, 9, 17, 18, 19, 20]
    ).astype(int)

    # 衍生特征工程

    # 1 行程时长（分钟）
    # 含义：反映出行时间成本，可用于需求分析
    # 作用：反映道路拥堵程度和出行时间成本，是分析出行效率和预测交通需求的重要特征。
    df_clean["trip_duration_min"] = (
                                            df_clean[dropoff_col] - df_clean[pickup_col]
                                    ).dt.total_seconds() / 60

    # 2 单位距离时间效率
    # 含义：衡量交通拥堵或路线效率
    # 作用：衡量道路通行效率和交通拥堵程度。
    if "trip_distance" in df_clean.columns:
        df_clean["time_per_mile"] = df_clean["trip_duration_min"] / (df_clean["trip_distance"] + 1e-6)
    else:
        df_clean["time_per_mile"] = np.nan

    # 3 夜间出行特征
    # 含义：夜间行为模式与白天不同（安全/需求/价格）
    # 作用：区分夜间与白天的出行行为差异，有助于分析不同时间段的需求特征和费用规律。
    df_clean["is_night"] = df_clean["pickup_hour"].isin(
        list(range(0, 6)) + list(range(22, 24))
    ).astype(int)

    # 输出清洗后的数据
    df_clean.to_csv("outputs/cleaned_data_sample.csv", index=False)
    print("处理完成：数据质量报告 + 清洗数据已生成")


    return df_clean