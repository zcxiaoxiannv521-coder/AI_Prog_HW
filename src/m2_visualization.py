import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os


OUTPUT_PATH = "outputs"
SHAPE_PATH = "data/taxi_zones.shp"


def visualization(df):

    os.makedirs(OUTPUT_PATH,exist_ok=True)
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 出行需求时间规律
    hourly_order = df.groupby("pickup_hour").size()
    plt.figure(figsize=(8,5) )
    plt.plot(
        hourly_order.index,
        hourly_order.values,
        marker="o",
        label="订单数量"
    )
    plt.xlabel("小时")
    plt.ylabel("订单数量")
    plt.title("全天不同时段出租车订单需求变化")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_PATH,"m2_1_demand.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 2. 区域热度分析（TOP10 + 分级设色地图）
    plot_region_analysis(df)

    # 3. 车费影响因素分析
    sample = df.sample(min(5000,len(df)),random_state=42)
    plt.figure(figsize=(8,5))
    plt.scatter(
        sample["trip_distance"],
        sample["fare_amount"],
        alpha=0.3,
        label="出租车订单"
    )
    plt.xlabel("行程距离(mile)")
    plt.ylabel("车费金额($)")
    plt.title("行程距离与出租车费用关系")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_PATH,"m2_3_fare.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 4. 自定义分析
    peak_fare = df.groupby("is_peak_hour")["fare_amount"].mean()
    plt.figure(figsize=(7, 5))
    plt.bar(
        ["非高峰", "高峰"],
        peak_fare.values,
        label="平均车费"
    )
    plt.ylabel("平均车费（$）")
    plt.title("高峰与非高峰平均车费对比")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_PATH, "m2_4_custom.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print("M2可视化完成，图片已保存")

def plot_region_analysis(df):

    if "PULocationID" not in df.columns:
        print("未找到PULocationID字段")
        return
    if not os.path.exists(SHAPE_PATH):
        print("未找到taxi_zones.shp")
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16,7)
    )

    region_count = (
        df["PULocationID"]
        .value_counts()
        .head(10)
        .sort_values()
    )


    axes[0].barh(
        region_count.index.astype(str),
        region_count.values,
        label="订单数量"
    )


    axes[0].set_title("出租车上车区域TOP10热度分析")
    axes[0].set_xlabel("订单数量")
    axes[0].set_ylabel("区域编号")
    axes[0].legend()
    axes[0].grid(
        axis="x",
        alpha=0.3
    )

    zones = gpd.read_file(SHAPE_PATH)
    pickup_count = (
        df.groupby("PULocationID")
        .size()
        .reset_index(
            name="pickup_count"
        )
    )


    map_df = zones.merge(
        pickup_count,
        left_on="LocationID",
        right_on="PULocationID",
        how="left"
    )
    map_df["pickup_count"] = (map_df["pickup_count"].fillna(0))
    map_df.plot(
        column="pickup_count",
        cmap="OrRd",
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        scheme="quantiles",
        k=5,
        ax=axes[1],
        legend_kwds={
            "title":"上车订单量分级"
        }
    )

    axes[1].set_title("纽约出租车上车区域空间热度分布")
    axes[1].axis("off")
    plt.suptitle(
        "纽约市出租车区域热度分析",
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "m2_2_region.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()