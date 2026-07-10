import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error)
# pytorch
import torch
import torch.nn as nn


OUTPUT_PATH = "outputs"

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 构造预测数据
def build_features(df):
    """
        构造出租车需求预测模型输入特征

        预测目标：
        demand（某区域某时间段内的出租车订单数量）

        输入特征：
        1. PULocationID
        2. pickup_hour
        3. pickup_weekday
        4. is_weekend
        5. is_peak_hour

        这些特征均来自出租车订单的时空属性,可以反映不同区域、不同时间条件下的出行需求变化规律。
        """
    demand = (df.groupby(["PULocationID","pickup_hour","pickup_weekday","is_weekend","is_peak_hour"]).size().reset_index(name="demand"))
    '''输入特征X
    (一)选择PULocationID的原因：
    1.不同区域具有不同的商业、居住和交通属性，
    2.出租车需求存在明显空间差异。
    (二)选择pickup_hour的原因：
    1.出租车需求具有明显时间周期性，例如早晚高峰和夜间需求存在较大差异。
    (三)选择pickup_weekday的原因：
    1.工作日和周末居民出行目的不同，会导致出租车订单规模变化。
    (三)选择is_weekend的原因：
    1.周末通常具有不同的出行模式，
    2.帮助模型学习非工作日需求变化。
    (四)选择is_peak_hour的原因：
    1.高峰时段交通需求通常更高，
    2.能够增强模型对特殊时间段需求变化的识别能力。'''
    X = demand.drop("demand",axis=1)
    '''预测目标y：
    demand表示某区域某小时内产生的订单数量，是衡量出租车出行需求的重要指标。'''
    y = demand["demand"]
    return X, y

# 神经网络模型
class DemandNN(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.network(x)

# 训练神经网络
def train_nn(
        X_train,
        y_train
):

    model = DemandNN(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    losses=[]
    epochs = 100
    for epoch in range(epochs):
        prediction = model(X_train)
        loss = criterion(prediction,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

# 主函数
def modeling(df):
    os.makedirs(OUTPUT_PATH,exist_ok=True)
    set_seed(42)

    # 数据准备
    X,y = build_features(df)
    # 8:2划分,按照80%训练集、20%测试集划分数据。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 转Tensor
    X_train_tensor = torch.tensor(X_train_scaled,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1,1),dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled,dtype=torch.float32)

    # 1. 神经网络训练
    '''神经网络通过多层非线性变换学习输入特征与出租车需求之间的复杂关系。
    在出租车需求预测任务中：
        优点：
        1.可以自动学习区域、时间等因素之间的非线性交互关系；
        2.随着数据规模增加，具有更强的拟合能力；
        3.适合处理出租车这种具有复杂时空变化规律的问题。
        缺点:
        1.模型参数较多，需要较长训练时间；
        2.对数据规模和参数设置较敏感；
        3.模型解释性较弱，难以直接说明某个因素的重要程度。'''
    nn_model, losses = train_nn(X_train_tensor,y_train_tensor)
    # 测试预测
    nn_model.eval()
    with torch.no_grad():
        nn_pred = nn_model(X_test_tensor).numpy()
    nn_pred = nn_pred.flatten()
    nn_mae = mean_absolute_error(y_test,nn_pred)
    nn_rmse = np.sqrt(mean_squared_error(y_test,nn_pred))

    # 保存loss曲线
    plt.figure(figsize=(8,5))
    plt.plot(losses,label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Neural Network Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_PATH,"m3_neural_network_loss.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 2. 随机森林
    '''随机森林由多个决策树组成，通过集成多个弱模型提高预测稳定性。
    在出租车需求预测任务中：
        优点：
        1.能够处理区域、时间等非线性影响因素；
        2.对异常数据具有较好的鲁棒性；
        3.不需要大量数据预处理，对特征尺度不敏感；
        4.可以通过特征重要性分析影响需求的关键因素。
        缺点：
        1.模型复杂度较高，预测速度相比简单模型较慢；
        2.对连续变化趋势的学习能力有限；
        3.当数据规模非常大时，存储和计算成本较高。'''
    rf = RandomForestRegressor(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test,rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test,rf_pred))

    # 保存模型指标
    result = pd.DataFrame(
        {
            "model":["Neural Network","Random Forest"],
            "MAE":[nn_mae,rf_mae],
            "RMSE":[nn_rmse,rf_rmse]

        }

    )
    result.to_csv(os.path.join(OUTPUT_PATH,"m3_model_metrics.csv"),index=False)
    print("M3预测模型完成")
    print(result)