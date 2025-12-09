
import time
import os
import copy
import gymnasium as gym
import env
import usv_ppo
import torch
import numpy as np


def get_usv_validate_env(env_paras):
    """
    创建USV验证环境

    基于validate.py的get_validate_env()函数，适配USV数据结构

    Args:
        env_paras: 环境参数，包含num_tasks、num_usvs、batch_size等

    Returns:
        env: USV验证环境实例
    """
    # 构建USV验证数据路径：./usv_data_dev/{num_tasks}_{num_usvs}/
    file_path = "./usv_data_dev/{0}_{1}/".format(
        env_paras["num_tasks"],
        env_paras["num_usvs"]
    )

    # 获取验证数据文件列表
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path + valid_data_files[i]

    # 创建USV验证环境，使用gymnasium
    env = gym.make('usv-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env


def usv_validate(env_paras, env, model_policy, extra_metrics=False):
    """
    USV策略验证函数

    基于validate.py的validate()函数，适配USV约束验证和组件

    Args:
        env_paras: 环境参数
        env: USV环境实例
        model_policy: 训练的策略模型
        extra_metrics: 是否记录额外指标（默认False，留作未来扩展）

    Returns:
        makespan: 平均最大完成时间
        makespan_batch: 每个batch的makespan
        extra_info: 额外信息字典（当extra_metrics=True时使用）
    """
    start = time.time()
    batch_size = env_paras["batch_size"]

    # 使用USV专用的Memory组件
    memory = usv_ppo.Memory()
    print('There are {0} dev instances.'.format(batch_size))

    # 获取初始状态
    state = env.state
    done = False
    dones = np.zeros(batch_size, dtype=bool)

    # 执行策略验证循环
    while not done:
        with torch.no_grad():
            # 使用USV策略模型执行动作选择
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        # 环境推进
        _, _, _, _, infos = env.step(actions)
        state = env.state

        dones_np = np.asarray(infos["batch_terminated"], dtype=bool) | np.asarray(infos["batch_truncated"], dtype=bool)
        done = bool(dones_np.all())
        dones = dones_np
    # USV约束验证（替代FJSP的gantt验证）
    validation_result = env.validate_constraints()
    is_valid = validation_result[0]

    if not is_valid:
        print("USV约束违反！！！！！！")

    # 计算makespan指标（与FJSP保持一致）
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)

    # 预留扩展接口（当前返回空字典）
    extra_info = {}
    if extra_metrics:
        extra_info = {
            "charging_stats": None,      # 充电统计（待实现）
            "usv_utilization": None,     # USV利用率（待实现）
            "constraint_violations": None, # 约束违反详情（待实现）
            "validation_details": validation_result[1] if len(validation_result) > 1 else None
        }

    # 重置环境
    env.reset()
    print('validating time: ', time.time() - start, '\n')

    return makespan, makespan_batch, extra_info


# 测试函数（可选）
def test_usv_validate():
    """
    USV验证模块测试函数
    用于验证模块功能的正确性
    """
    print("=== USV验证模块测试 ===")

    # 测试环境参数
    test_env_paras = {
        "num_tasks": 40,
        "num_usvs": 4,
        "batch_size": 20,
        "valid_batch_size": 100
    }

    try:
        # 测试环境创建
        print("1. 测试USV验证环境创建...")
        env = get_usv_validate_env(test_env_paras)
        print("   [PASS] 环境创建成功")

        # 测试验证函数（需要已训练的模型）
        print("2. 测试USV验证函数...")
        print("   注意：需要已训练的USV模型进行完整测试")

        return True

    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    test_usv_validate()
