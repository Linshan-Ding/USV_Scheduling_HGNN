"""
USV模型保存管理器

提供配置化的模型保存、加载和管理功能
支持实验版本控制、检查点管理、模型对比等功能
"""

import os
import json
import torch
import shutil
import time
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pandas as pd


class SaveManager:
    """
    USV模型保存管理器

    功能：
    - 配置化的实验目录创建和管理
    - 最佳模型保存和管理（支持多种指标）
    - 训练检查点保存和恢复
    - 配置文件备份和版本管理
    - 实验结果记录和查询
    """

    def __init__(self, config: Dict):
        """
        初始化保存管理器

        Args:
            config: 包含save_paras的配置字典
        """
        self.save_config = config.get("save_paras", {})
        self.env_paras = config.get("env_paras", {})

        # 解析配置参数
        self.base_path = self.save_config.get("base_path", "./save")
        self.experiment_name = self.save_config.get("experiment_name", "usv_experiment")
        self.experiment_version = self.save_config.get("experiment_version", "v1.0")
        self.auto_version = self.save_config.get("auto_version", True)
        self.max_best_models = self.save_config.get("max_best_models", 3)
        self.save_interval = self.save_config.get("save_interval", 10)
        self.checkpoint_format = self.save_config.get("checkpoint_format", "pt")
        self.backup_config = self.save_config.get("backup_config", True)

        # 生成实验目录
        self.experiment_dir = self._setup_experiment_dir()

        # 初始化目录结构
        self._create_directory_structure()

        # 最佳模型管理
        self.best_models = {
            "makespan": deque(maxlen=self.max_best_models),
            "reward": deque(maxlen=self.max_best_models)
        }

        # 备份当前配置
        if self.backup_config:
            self._backup_config(config)

    def _setup_experiment_dir(self) -> str:
        """
        设置实验目录

        Returns:
            实验根目录路径
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 构建实验名称
        if self.auto_version:
            exp_name = f"{self.experiment_name}_{self.experiment_version}_{timestamp}"
        else:
            exp_name = self.save_config.get("naming_pattern", "{exp_name}_{timestamp}").format(
                exp_name=self.experiment_name,
                task_count=self.env_paras.get("num_tasks", "N"),
                usv_count=self.env_paras.get("num_usvs", "M"),
                timestamp=timestamp
            )

        experiment_dir = os.path.join(self.base_path, exp_name)
        return experiment_dir

    def _create_directory_structure(self):
        """创建实验目录结构"""
        directories = [
            "models",           # 最佳模型
            "models/checkpoints",  # 训练检查点
            "logs",            # 训练日志
            "config",          # 配置备份
            "results",         # 结果文件
            "analysis"         # 分析结果
        ]

        for dir_name in directories:
            dir_path = os.path.join(self.experiment_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)

        print(f"[SaveManager] 实验目录已创建: {self.experiment_dir}")

    def _backup_config(self, config: Dict):
        """
        备份配置文件到实验目录

        Args:
            config: 要备份的配置字典
        """
        config_backup_path = os.path.join(
            self.experiment_dir, "config",
            f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(config_backup_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"[SaveManager] 配置已备份: {config_backup_path}")

    def get_model_path(self, model_type: str = "best", metric: str = "makespan",
                      extra_info: str = "") -> str:
        """
        获取模型保存路径

        Args:
            model_type: 模型类型 ("best", "checkpoint", "final")
            metric: 评估指标 ("makespan", "reward")
            extra_info: 额外信息

        Returns:
            模型文件路径
        """
        if model_type == "best":
            filename = f"best_{metric}"
            if extra_info:
                filename += f"_{extra_info}"
        elif model_type == "checkpoint":
            filename = f"checkpoint_{extra_info}"
        elif model_type == "final":
            filename = f"final_model"
        else:
            filename = model_type

        filename += f".{self.checkpoint_format}"
        return os.path.join(self.experiment_dir, "models", filename)

    def save_best_model(self, model, metrics: Dict[str, float],
                       model_info: Dict = None, metric_type: str = "makespan"):
        """
        保存最佳模型

        Args:
            model: 要保存的模型
            metrics: 评估指标字典
            model_info: 模型额外信息
            metric_type: 主要优化指标类型
        """
        metric_value = metrics.get(metric_type)
        if metric_value is None:
            print(f"[SaveManager] 警告: 指标 {metric_type} 不存在")
            return

        # 生成模型信息
        model_info = model_info or {}
        model_info.update({
            "metrics": metrics,
            "save_time": datetime.now().isoformat(),
            "metric_type": metric_type,
            "experiment_dir": self.experiment_dir
        })

        # 生成文件路径
        task_count = self.env_paras.get("num_tasks", "N")
        usv_count = self.env_paras.get("num_usvs", "M")
        timestamp = datetime.now().strftime("%H%M%S")
        extra_info = f"{task_count}x{usv_count}_{timestamp}"

        model_path = self.get_model_path("best", metric_type, extra_info)

        # 保存模型（支持不同的模型结构）
        if hasattr(model, 'hgnn_scheduler_old') and hasattr(model, 'mlps_old'):
            # USV PPO模型结构
            model_data = {
                'hgnn_scheduler': model.hgnn_scheduler_old.state_dict(),
                'mlps': model.mlps_old.state_dict(),
                'model_info': model_info
            }
        elif hasattr(model, 'state_dict'):
            # 标准PyTorch模型
            model_data = {
                'model_state_dict': model.state_dict(),
                'model_info': model_info
            }
        else:
            print(f"[SaveManager] 错误: 不支持的模型结构")
            return

        # 保存模型文件
        torch.save(model_data, model_path)

        # 更新最佳模型列表
        self.best_models[metric_type].append({
            "path": model_path,
            "metric_value": metric_value,
            "metrics": metrics,
            "save_time": model_info["save_time"]
        })

        # 清理旧模型（如果超过最大数量）
        self._cleanup_old_models(metric_type)

        print(f"[SaveManager] 最佳模型已保存: {model_path}")
        print(f"[SaveManager] {metric_type}: {metric_value:.4f}")

    def save_checkpoint(self, model, optimizer, epoch: int,
                       loss_info: Dict = None, **kwargs):
        """
        保存训练检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            loss_info: 损失信息
            **kwargs: 其他要保存的信息
        """
        checkpoint_path = os.path.join(
            self.experiment_dir, "models", "checkpoints",
            f"checkpoint_epoch_{epoch}.{self.checkpoint_format}"
        )

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'save_time': datetime.now().isoformat(),
        }

        if loss_info:
            checkpoint_data['loss_info'] = loss_info

        # 添加额外信息
        checkpoint_data.update(kwargs)

        torch.save(checkpoint_data, checkpoint_path)
        print(f"[SaveManager] 检查点已保存: epoch {epoch}")

    def load_model(self, model_path: str, model=None):
        """
        加载模型

        Args:
            model_path: 模型文件路径
            model: 要加载到的模型（可选）

        Returns:
            加载的模型数据字典
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')

        if model is not None:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'hgnn_scheduler' in checkpoint and 'mlps' in checkpoint:
                # USV PPO模型结构
                model.hgnn_scheduler_old.load_state_dict(checkpoint['hgnn_scheduler'])
                model.mlps_old.load_state_dict(checkpoint['mlps'])
            else:
                print(f"[SaveManager] 警告: 未知的模型结构")

        print(f"[SaveManager] 模型已加载: {model_path}")
        return checkpoint

    def load_best_model(self, model, metric_type: str = "makespan"):
        """
        加载最佳模型

        Args:
            model: 要加载到的模型
            metric_type: 指标类型

        Returns:
            模型信息字典
        """
        if not self.best_models[metric_type]:
            raise ValueError(f"没有找到 {metric_type} 类型的最佳模型")

        best_model_info = self.best_models[metric_type][-1]  # 获取最新的最佳模型
        checkpoint = self.load_model(best_model_info["path"], model)

        return checkpoint.get('model_info', {})

    def save_training_log(self, log_data: Dict, log_name: str = "training_log"):
        """
        保存训练日志

        Args:
            log_data: 日志数据
            log_name: 日志文件名
        """
        log_path = os.path.join(self.experiment_dir, "logs", f"{log_name}.json")

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def save_results_csv(self, data: Dict, filename: str):
        """
        保存结果到CSV文件

        Args:
            data: 要保存的数据字典
            filename: 文件名
        """
        results_path = os.path.join(self.experiment_dir, "results", filename)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        df.to_csv(results_path, index=False)
        print(f"[SaveManager] 结果已保存: {results_path}")

    def get_experiment_summary(self) -> Dict:
        """
        获取实验总结信息

        Returns:
            实验总结字典
        """
        summary = {
            "experiment_dir": self.experiment_dir,
            "experiment_name": self.experiment_name,
            "config": self.save_config,
            "best_models": {},
            "saved_files": self._count_saved_files(),
            "creation_time": datetime.now().isoformat()
        }

        # 添加最佳模型信息
        for metric_type in self.best_models:
            if self.best_models[metric_type]:
                best = self.best_models[metric_type][-1]
                summary["best_models"][metric_type] = {
                    "path": best["path"],
                    "metric_value": best["metric_value"],
                    "save_time": best["save_time"]
                }

        return summary

    def _cleanup_old_models(self, metric_type: str):
        """
        清理旧的模型文件

        Args:
            metric_type: 指标类型
        """
        if len(self.best_models[metric_type]) > self.max_best_models:
            # 删除最旧的模型
            old_model = self.best_models[metric_type][0]
            if os.path.exists(old_model["path"]):
                os.remove(old_model["path"])
                print(f"[SaveManager] 已清理旧模型: {old_model['path']}")

    def _count_saved_files(self) -> Dict[str, int]:
        """
        统计保存的文件数量

        Returns:
            文件数量统计字典
        """
        file_counts = {}
        for root, dirs, files in os.walk(self.experiment_dir):
            category = os.path.relpath(root, self.experiment_dir)
            category = category if category != '.' else 'root'
            file_counts[category] = len(files)

        return file_counts

    def list_experiments(self, base_path: str = None) -> List[Dict]:
        """
        列出所有实验

        Args:
            base_path: 基础路径（可选）

        Returns:
            实验信息列表
        """
        base_path = base_path or self.base_path
        experiments = []

        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # 尝试读取实验信息
                config_dir = os.path.join(item_path, "config")
                if os.path.exists(config_dir):
                    config_files = glob.glob(os.path.join(config_dir, "config_backup_*.json"))
                    if config_files:
                        try:
                            with open(config_files[-1], 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                experiments.append({
                                    "name": item,
                                    "path": item_path,
                                    "config": config.get("save_paras", {}),
                                    "creation_time": datetime.fromtimestamp(
                                        os.path.getctime(item_path)
                                    ).isoformat()
                                })
                        except Exception as e:
                            print(f"[SaveManager] 读取实验配置失败: {e}")

        return sorted(experiments, key=lambda x: x["creation_time"], reverse=True)