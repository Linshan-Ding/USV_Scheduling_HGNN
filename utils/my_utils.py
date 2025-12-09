import json

# =============================================================================
# USV模型集中常量定义
# =============================================================================

# 模型维度常量
DEFAULT_EMBEDDING_DIM = 32  # 嵌入维度：与usv_hgnn.py保持一致
DEFAULT_HIDDEN_DIM = 128    # 隐藏层维度：等于4*d，避免信息瓶颈

# 网络结构常量
DEFAULT_ACTION_LAYERS = 3   # 动作头网络层数
DEFAULT_VALUE_LAYERS = 2    # 价值头网络层数

# 数值处理常量
MASK_FILL_VALUE = float('-inf')  # 掩码填充值：统一使用-inf替代-1e9，数学语义更清晰
NUMERICAL_STABILITY_EPS = 1e-8   # 数值稳定性：防止除零误差的小常数

# 训练常量
DEFAULT_MINIBATCH_SIZE = 64    # 默认minibatch大小


def read_json(path:str) -> dict:
    with open(path+".json","r",encoding="utf-8") as f:
        config = json.load(f)
    return config

def write_json(data:dict, path:str):
    with open(path+".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))