# NYC出租车数据分析与智能问答系统

## 1. 项目简介

本项目基于纽约市黄色出租车公开数据集（NYC Yellow Taxi Trip Records），
完成出租车数据的数据处理、可视化分析、需求预测以及智能问答功能。

项目主要包括以下模块：

- M1：数据读取与清洗
- M2：数据可视化分析
- M3：出租车需求预测模型
- M4：智能问答系统

## 项目特点

- 基于真实纽约出租车大规模数据；
- 完成数据清洗、分析、预测完整流程；
- 使用深度学习和传统机器学习模型进行需求预测；
- 引入空间数据分析方法，实现区域热度地图；
- 集成大语言模型，实现自然语言问答。

## 2. 项目环境

运行环境：

- Python 3.9+

requirements.txt中包含项目运行所需第三方库：

主要依赖包括：

- pandas：数据处理
- numpy：数值计算
- matplotlib：数据可视化
- geopandas：空间数据处理与地图绘制
- mapclassify：区域分级设色
- scikit-learn：随机森林模型
- torch：神经网络模型
- pyarrow：Parquet文件读取
- openai：大模型接口调用
- gradio：交互界面构建


## 3. 项目文件结构
```
final_project/

│
├── .venv/
│   └── ...
│
│
├── data/
│   ├── taxi_zones.cpg
│   ├── taxi_zones.dbf
│   ├── taxi_zones.prj
│   ├── taxi_zones.shp
│   ├── taxi_zones.shx
│   └── yellow_tripdata_2026-01.parquet
│
│
├── outputs/
│   ├── cleaned_data_sample.csv
│   ├── data_quality_report.csv
│   ├── m2_1_demand.png
│   ├── m2_2_region.png
│   ├── m2_3_fare.png
│   ├── m2_4_custom.png
│   ├── m3_model_metrics.csv
│   └── m3_neural_network_loss.png
│
│
├── src/
│   ├── llm_qa.py
│   ├── m1_data_processing.py
│   ├── m2_visualization.py
│   ├── m3_modeling.py
│   └── m4_qa_system.py
│
│
├── gradio_app.py
│
├── main.py
│
├── README.md
│
├── requirements.txt
│
└── 人机协作报告.md

```

## 4. 数据说明

本项目使用纽约市黄色出租车公开数据：

数据来源：

NYC Taxi & Limousine Commission (TLC)

主要字段包括：

- 行程开始时间
- 行程结束时间
- 行程距离
- 上下车区域
- 乘客数量
- 车费金额
- 支付方式


数据文件存放于：
```
final_project/
│
└── data/
    ├── taxi_zones.cpg
    ├── taxi_zones.dbf
    ├── taxi_zones.prj
    ├── taxi_zones.shp
    ├── taxi_zones.shx
    └── yellow_tripdata_2026-01.parquet
```


## 5. 安装依赖

在项目根目录打开终端，执行：

```bash
pip install -r requirements.txt
```

## 6. 项目运行方式
### （1）运行完整项目

在项目根目录下执行：

```bash
python main.py
```
### （2）运行 Gradio 可视化界面
在项目根目录下执行：
```bash
python gradio_app.py
```
## 7. 模块功能说明

### M1 数据读取与清洗模块

文件：src/m1_data_processing.py


主要功能：

- 读取Parquet格式出租车数据；
- 数据缺失率统计；
- 异常值检测；
- 数据清洗；
- 时间特征构造。


主要生成：
outputs/data_quality_report.csv

用于记录数据质量情况。



---

### M2 数据可视化模块

文件：src/m2_visualization.py


主要完成：

1. 出行需求时间规律分析

分析不同小时出租车订单变化。


2. 区域热度分析

包括：

（1）统计出租车上车区域TOP10订单量；

（2）利用geopandas加载taxi_zones.shp，
结合空间数据完成区域订单量分级设色地图。

输出：

m2_2_region.png

m2_bonus_zone_map.png


3. 费用影响因素分析

研究行程距离与车费之间关系。


4. 自定义分析

输出：

m2_1_demand.png

m2_2_region.png

m2_3_fare.png

m2_4_custom.png

---

### M3 需求预测模型模块

文件：src/m3_modeling.py

预测目标：

根据时间和区域信息预测出租车需求量。

输入特征：

- PULocationID
- pickup_hour
- pickup_weekday
- is_weekend
- is_peak_hour

模型：

1. 神经网络模型（PyTorch）

2. 随机森林模型（Scikit-learn）

评价指标：

- MAE
- RMSE

输出：

outputs/m3_model_metrics.csv

outputs/m3_neural_network_loss.png

---

### M4 智能问答模块

文件：src/m4_qa_system.py


支持：

- 时段需求查询
- 区域排名查询
- 费用查询
- 数据统计查询
- 模型评价查询


对于规则无法匹配的问题，
调用大模型进行解释性回答。

## 8. 输出文件说明


所有分析结果均保存在：outputs/


|文件|说明|
|-|-|
|cleaned_data_sample.csv|清洗后的出租车数据|
|data_quality_report.csv|数据质量分析报告|
|m2_1_demand.png|出租车时间需求变化图|
|m2_2_region.png|区域订单量TOP10分析图|
|m2_3_fare.png|距离与车费关系图|
|m2_4_custom.png|自定义分析结果|
|m3_model_metrics.csv|模型评价指标|
|m3_neural_network_loss.png|神经网络训练Loss曲线|

## 9. 注意事项


1. 运行前请确保数据文件已经放入：data/


2. taxi_zones.shp文件需要与以下文件同时存在：

taxi_zones.shp

taxi_zones.shx

taxi_zones.dbf

taxi_zones.prj

taxi_zones.cpg

3. 使用大模型问答功能时，需要提前配置API Key。


4. Gradio运行成功后：

终端会显示访问地址：

http://127.0.0.1:7860

在浏览器打开即可进入智能问答界面。