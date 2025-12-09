"""
USV模型工具模块

提供模型加载、评估、比较等功能
与usv_validate.py配合使用，专注于模型生命周期管理
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

import usv_ppo
from usv_validate import get_usv_validate_env, usv_validate
from utils.my_utils import read_json
from utils.save_manager import SaveManager


def load_model_from_checkpoint(checkpoint_path: str, model_paras: dict, train_paras: dict, batch_size: int = 20):
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        model_paras: 模型参数
        train_paras: 训练参数
        batch_size: 批次大小

    Returns:
        加载的模型实例
    """
    print(f"[model_tools] 正在加载模型: {checkpoint_path}")

    # 创建新模型实例
    model = usv_ppo.PPO(model_paras, train_paras, num_envs=batch_size)

    # 使用SaveManager加载模型
    save_manager = SaveManager({"save_paras": {}})
    checkpoint = save_manager.load_model(checkpoint_path, model)

    print("[model_tools] 模型加载完成")
    return model


def find_best_model_path(experiment_dir: str):
    """
    在实验目录中查找最佳模型

    Args:
        experiment_dir: 实验目录路径

    Returns:
        最佳模型文件路径，如果未找到则返回None
    """
    models_dir = os.path.join(experiment_dir, "models")
    if not os.path.exists(models_dir):
        print(f"[model_tools] 模型目录不存在: {models_dir}")
        return None

    # 查找最佳makespan模型
    best_models = [f for f in os.listdir(models_dir) if f.startswith("best_makespan_")]
    if not best_models:
        print(f"[model_tools] 在 {experiment_dir} 中未找到最佳模型")
        return None

    # 返回最新的最佳模型
    best_models.sort()  # 按文件名排序，通常包含时间戳
    model_path = os.path.join(models_dir, best_models[-1])
    print(f"[model_tools] 找到最佳模型: {model_path}")
    return model_path


def load_and_evaluate_model(config_path: str, model_path: str = None, experiment_dir: str = None):
    """
    加载模型并评估性能

    Args:
        config_path: 配置文件路径
        model_path: 模型文件路径（可选）
        experiment_dir: 实验目录路径（可选，用于加载最佳模型）

    Returns:
        评估结果字典
    """
    print("[model_tools] 开始模型加载和评估...")

    # 加载配置
    config = read_json(config_path)
    env_paras = config["env_paras"]
    model_paras = config["model_paras"]
    train_paras = config["train_paras"]

    # 确定模型路径
    if model_path is None and experiment_dir is not None:
        model_path = find_best_model_path(experiment_dir)

    if model_path is None:
        raise ValueError("必须提供model_path或experiment_dir参数")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型
    batch_size = env_paras.get("valid_batch_size", env_paras.get("batch_size", 20))
    model = load_model_from_checkpoint(model_path, model_paras, train_paras, batch_size)

    # 创建验证环境
    env_valid_paras = env_paras.copy()
    env_valid_paras["batch_size"] = batch_size
    env_valid = get_usv_validate_env(env_valid_paras)

    # 模型包装器（适配验证接口）
    class USVPolicyWrapper:
        def __init__(self, hgnn_scheduler, mlps):
            self.hgnn_scheduler = hgnn_scheduler
            self.mlps = mlps

        def act(self, state, memory, dones, flag_sample=False, flag_train=False):
            deterministic = not flag_sample
            actions, logprobs, _ = self.hgnn_scheduler.select_action(state, deterministic)
            return actions

    policy_wrapper = USVPolicyWrapper(model.hgnn_scheduler_old, model.mlps_old)

    # 执行评估
    print("[model_tools] 开始模型评估...")
    makespan_avg, makespan_instances, _ = usv_validate(env_valid_paras, env_valid, policy_wrapper)

    # 构建结果
    results = {
        "model_path": model_path,
        "config_path": config_path,
        "experiment_dir": experiment_dir,
        "evaluation_time": datetime.now().isoformat(),
        "makespan_avg": makespan_avg.item() if hasattr(makespan_avg, 'item') else makespan_avg,
        "task_count": env_paras["num_tasks"],
        "usv_count": env_paras["num_usvs"],
        "valid_batch_size": batch_size,
        "makespan_instances": makespan_instances.tolist() if hasattr(makespan_instances, 'tolist') else makespan_instances
    }

    # 输出结果
    print("\n" + "="*60)
    print("评估结果:")
    print(f"模型路径: {model_path}")
    print(f"平均makespan: {results['makespan_avg']:.4f}")
    print(f"任务数量: {results['task_count']}")
    print(f"USV数量: {results['usv_count']}")
    print(f"验证批次大小: {results['valid_batch_size']}")
    print("="*60)

    # 保存评估结果
    if experiment_dir:
        output_dir = experiment_dir
    else:
        output_dir = os.path.dirname(model_path) if model_path else "./results"
        os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[model_tools] 评估结果已保存: {results_path}")
    return results


def compare_experiments(config_path: str, experiment_dirs: list):
    """
    比较多个实验的结果

    Args:
        config_path: 配置文件路径
        experiment_dirs: 实验目录列表

    Returns:
        比较结果列表
    """
    print("[model_tools] 开始比较实验结果...")

    comparison_results = []

    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n[{i}/{len(experiment_dirs)}] 评估实验: {exp_dir}")

        try:
            result = load_and_evaluate_model(config_path, experiment_dir=exp_dir)
            result["experiment_dir"] = exp_dir
            comparison_results.append(result)
        except Exception as e:
            print(f"[model_tools] 评估实验 {exp_dir} 时出错: {e}")
            continue

    if not comparison_results:
        print("[model_tools] 没有成功评估的实验")
        return []

    # 创建比较表格
    df = pd.DataFrame(comparison_results)

    # 按makespan排序
    df = df.sort_values('makespan_avg')

    print("\n" + "="*80)
    print("实验比较结果:")
    print("="*80)
    print(df[['experiment_dir', 'makespan_avg', 'task_count', 'usv_count']].to_string(index=False))
    print("="*80)

    # 保存比较结果
    output_path = f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    print(f"[model_tools] 比较结果已保存: {output_path}")

    return comparison_results


def list_saved_experiments(base_path: str = "./save"):
    """
    列出所有保存的实验

    Args:
        base_path: 基础保存路径

    Returns:
        实验信息列表
    """
    print(f"[model_tools] 列出保存路径: {base_path}")

    save_manager = SaveManager({"save_paras": {"base_path": base_path}})
    experiments = save_manager.list_experiments(base_path)

    if not experiments:
        print("[model_tools] 未找到任何实验")
        return []

    print("\n" + "="*80)
    print("保存的实验:")
    print("="*80)

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}] {exp['name']}")
        print(f"    路径: {exp['path']}")
        print(f"    创建时间: {exp['creation_time']}")

        # 尝试读取最佳模型信息
        best_model_path = find_best_model_path(exp['path'])
        if best_model_path:
            print(f"    最佳模型: {os.path.basename(best_model_path)}")
        else:
            print("    最佳模型: 未找到")

        print()

    print("="*80)
    return experiments


if __name__ == "__main__":
    # 如果直接运行此文件，提供简单的命令行接口
    import argparse

    parser = argparse.ArgumentParser(description='USV模型工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估单个模型')
    eval_parser.add_argument('--config', required=True, help='配置文件路径')
    eval_parser.add_argument('--model', help='模型文件路径')
    eval_parser.add_argument('--experiment', help='实验目录路径')

    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较多个实验')
    compare_parser.add_argument('--config', required=True, help='配置文件路径')
    compare_parser.add_argument('--experiments', nargs='+', required=True, help='实验目录列表')

    # 列表命令
    list_parser = subparsers.add_parser('list', help='列出保存的实验')
    list_parser.add_argument('--path', default='./save', help='保存基础路径')

    args = parser.parse_args()

    if args.command == 'evaluate':
        if not args.model and not args.experiment:
            print("错误: 必须提供 --model 或 --experiment 参数")
        else:
            load_and_evaluate_model(args.config, args.model, args.experiment)

    elif args.command == 'compare':
        compare_experiments(args.config, args.experiments)

    elif args.command == 'list':
        list_saved_experiments(args.path)

    else:
        parser.print_help()