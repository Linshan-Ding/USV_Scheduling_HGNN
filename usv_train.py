
import copy
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import usv_ppo  # USV专用PPO算法
from env.usv_case_generator import USVCaseGenerator  # USV案例生成器
from env.usv_env import USVState, unwrap_env, get_env_attr  # USV状态类与属性访问工具
from usv_validate import usv_validate, get_usv_validate_env  # USV验证模块
from utils.my_utils import read_json  # 工具函数
from utils.save_manager import SaveManager  # 新的保存管理器


def numpy_obs_to_state(env, observation):
    """
    将numpy观察转换为USVState对象（用于兼容PPO模型）

    Args:
        env: USV环境实例
        observation: numpy观察数组

    Returns:
        USVState: 重构的状态对象
    """
    import torch

    core_env = unwrap_env(env)

    # 将numpy观察转换为torch张量
    if isinstance(observation, np.ndarray):
        obs_tensor = torch.from_numpy(observation).to(core_env.device)
    else:
        obs_tensor = observation

    batch_size = obs_tensor.shape[0]

    # 解析观察向量，重构状态特征
    total_dim = obs_tensor.shape[1]

    # 计算各部分维度（与_state_to_numpy方法对应）
    task_feat_total = core_env.task_feat_dim * core_env.num_tasks
    usv_feat_total = core_env.usv_feat_dim * core_env.num_usvs
    adj_matrix_total = core_env.num_tasks * core_env.num_usvs
    mask_total = core_env.num_tasks + core_env.num_usvs

    # 分割观察向量
    current_idx = 0

    # 任务特征
    task_feat_end = current_idx + task_feat_total
    task_features_flat = obs_tensor[:, current_idx:task_feat_end]
    feat_tasks_batch = task_features_flat.view(batch_size, core_env.task_feat_dim, core_env.num_tasks)
    current_idx = task_feat_end

    # USV特征
    usv_feat_end = current_idx + usv_feat_total
    usv_features_flat = obs_tensor[:, current_idx:usv_feat_end]
    feat_usvs_batch = usv_features_flat.view(batch_size, core_env.usv_feat_dim, core_env.num_usvs)
    current_idx = usv_feat_end

    # 邻接矩阵
    adj_end = current_idx + adj_matrix_total
    adj_flat = obs_tensor[:, current_idx:adj_end]
    task_usv_adj_batch = adj_flat.view(batch_size, core_env.num_tasks, core_env.num_usvs).long()
    current_idx = adj_end

    # 掩码
    mask_end = current_idx + mask_total
    mask_flat = obs_tensor[:, current_idx:mask_end]
    task_finish_batch = mask_flat[:, :core_env.num_tasks].bool()
    usv_proc_batch = mask_flat[:, core_env.num_tasks:].bool()

    # 时间信息
    time_batch = obs_tensor[:, -1]

    # 创建state对象
    state = USVState(
        batch_idxes=torch.arange(batch_size, device=core_env.device),
        feat_tasks_batch=feat_tasks_batch,
        feat_usvs_batch=feat_usvs_batch,
        proc_times_batch=core_env.proc_times_batch,
        task_usv_adj_dynamic_batch=task_usv_adj_batch,
        task_usv_adj_batch=task_usv_adj_batch,
        task_types_batch=core_env.task_types_batch,
        end_task_biases_batch=core_env.end_task_biases_batch,
        nums_tasks_batch=core_env.nums_tasks_batch,
        mask_task_procing_batch=torch.zeros_like(task_finish_batch),
        mask_task_finish_batch=task_finish_batch,
        mask_usv_procing_batch=~usv_proc_batch,
        task_step_batch=core_env.task_step_batch,
        time_batch=time_batch
    )

    # 设置充电管理器引用（类变量，不在初始化中传递）
    state.charging_manager = core_env.charging_manager

    return state


def setup_seed(seed):
    """
    设置随机种子确保实验可复现性

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """
    USV训练主函数
    """
    # === 1. PyTorch初始化和设备配置 ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None,
                          linewidth=None, profile=None, sci_mode=False)

    # === 2. 加载USV配置文件和参数解析 ===
    config = read_json("./config_usv")  # 使用USV专用配置
    env_paras = config["env_paras"]
    model_paras = config["model_paras"]
    train_paras = config["train_paras"]

    # 设置设备
    env_paras["device"] = device
    model_paras["device"] = device

    # 验证环境参数（使用更大的batch size进行验证）
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]

    # USV模型维度计算（基于USV的二维动作空间）
    # USV动作为(task_id, usv_id)对，所以输入维度需要考虑任务和USV特征
    model_paras["actor_in_dim"] = model_paras["out_size_usv"] * 2 + model_paras["out_size_task"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_usv"] + model_paras["out_size_task"]

    # === 3. 创建USV专用训练对象 ===
    num_tasks = env_paras["num_tasks"]
    num_usvs = env_paras["num_usvs"]

    # 创建PPO Memory和模型
    memories = usv_ppo.Memory()
    model = usv_ppo.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])

    # 创建验证环境
    env_valid = get_usv_validate_env(env_valid_paras)

    # === 4. 创建保存管理器 ===
    save_manager = SaveManager(config)
    makespan_best = float('inf')

    # === 5. 可视化支持（visdom） ===
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # === 6. 初始化训练数据存储 ===
    valid_results = []
    valid_results_100 = []
    training_log = {
        "config": config,
        "start_time": time.time(),
        "iterations": [],
        "makespan_avg": [],
        "makespan_100": [],
        "best_models": []
    }

    # === 6. 开始训练迭代 ===
    start_time = time.time()
    env = None

    for i in range(1, train_paras["max_iterations"]+1):
        # 每 parallel_iter 步替换训练实例
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # === 案例：USVCaseGenerator集成 ===
            case = USVCaseGenerator(num_usvs, num_tasks)  # 使用USV案例生成器，注意参数顺序：先USV数量，后任务数量
            # === 环境：gymnasium.make('usv-v0')集成 ===
            env = gym.make('usv-v0', case=case, env_paras=env_paras)
            print('num_tasks: ', num_tasks, '\tnum_usvs: ', num_usvs)

        # 使用新的gymnasium API reset方法获取初始状态
        observations, info = env.reset()
        # 将numpy观察转换为USVState对象以兼容PPO模型
        state = numpy_obs_to_state(env, observations)
        done = False
        last_time = time.time()

        # === 单次迭代循环 ===
        while not done:
            with torch.no_grad():
                actions, logprobs, _ = model.select_action(state, deterministic=False)
                # 存储到memory
                memories.states.append(state)
                memories.logprobs.append(logprobs)
                memories.actions.append(actions)
            # 使用新的gymnasium API返回5元组 (obs, reward, terminated, truncated, info)
            observations, _, _, _, infos = env.step(actions)

            batch_rewards_np = np.asarray(infos["batch_rewards"], dtype=np.float64)
            batch_terminated_np = np.asarray(infos["batch_terminated"], dtype=bool)
            batch_truncated_np = np.asarray(infos["batch_truncated"], dtype=bool)
            dones_np = batch_terminated_np | batch_truncated_np

            done = bool(dones_np.all())
            # 将numpy观察转换为USVState对象（用于兼容PPO模型）
            state = numpy_obs_to_state(env, observations)

            rewards_tensor = torch.from_numpy(batch_rewards_np).to(torch.float32)
            dones_tensor = torch.from_numpy(dones_np.astype(np.bool_))

            memories.rewards.append(rewards_tensor)
            memories.is_terminals.append(dones_tensor)

        # 验证解决方案（使用USV约束验证）
        validation_result = env.validate_constraints()
        if not validation_result[0]:
            print("USV约束违反！！！！！！")
        observations, info = env.reset()  # 新API返回(obs, info)元组

        # === 策略更新和验证流程 ===
        # 每 update_timestep 步更新策略
        if i % train_paras["update_timestep"] == 0:
            # USV PPO的update方法需要optimizer和memory
            loss_dict = model.update(model.optimizer, memories)
            loss = loss_dict.get("loss", 0.0)
            reward = 0.0  # USV PPO的update不直接返回reward，可以从memory中计算平均奖励
            if len(memories.rewards) > 0:
                all_rewards = torch.cat(memories.rewards, dim=0)
                reward = all_rewards.mean().item()

            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of usv_envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of usv_envs'))

        # 每 save_timestep 步验证策略并保存最佳模型
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # 记录平均结果和每个实例的结果
            # 创建一个临时的策略对象来适配usv_validate的接口
            class USVPolicyWrapper:
                def __init__(self, hgnn_scheduler, mlps):
                    self.hgnn_scheduler = hgnn_scheduler
                    self.mlps = mlps

                def act(self, state, memory, dones, flag_sample=False, flag_train=False):
                    """适配FJSP风格的act方法调用"""
                    deterministic = not flag_sample
                    actions, logprobs, _ = self.hgnn_scheduler.select_action(state, deterministic)
                    return actions

            policy_wrapper = USVPolicyWrapper(model.hgnn_scheduler_old, model.mlps_old)
            vali_result, vali_result_100, _ = usv_validate(env_valid_paras, env_valid, policy_wrapper)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)

            # === 模型保存和最佳模型管理 ===
            # 更新训练日志
            training_log["iterations"].append(i)
            training_log["makespan_avg"].append(vali_result.item())

            # 使用SaveManager保存最佳模型
            metrics = {
                "makespan": vali_result.item(),
                "iteration": i,
                "timestamp": time.time()
            }

            model_info = {
                "num_tasks": num_tasks,
                "num_usvs": num_usvs,
                "iteration": i,
                "training_time": time.time() - start_time
            }

            if vali_result < makespan_best:
                makespan_best = vali_result
                save_manager.save_best_model(model, metrics, model_info, "makespan")
                training_log["best_models"].append({
                    "iteration": i,
                    "makespan": vali_result.item(),
                    "save_time": time.time()
                })

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of usv_valid'))

    # === 7. 保存训练结果 ===
    print("训练完成，正在保存结果...")

    # 保存最终模型
    final_metrics = {
        "makespan": makespan_best.item() if hasattr(makespan_best, 'item') else makespan_best,
        "total_iterations": len(training_log["iterations"]),
        "total_time": time.time() - start_time
    }
    save_manager.save_best_model(model, final_metrics, model_info, "makespan")

    # 保存训练日志
    training_log["end_time"] = time.time()
    training_log["total_time"] = training_log["end_time"] - training_log["start_time"]
    training_log["best_makespan"] = makespan_best.item() if hasattr(makespan_best, 'item') else makespan_best
    save_manager.save_training_log(training_log, "full_training_log")

    # 保存CSV格式的结果（便于分析）
    results_data = {
        "iteration": training_log["iterations"],
        "makespan_avg": training_log["makespan_avg"]
    }
    save_manager.save_results_csv(results_data, "training_curves.csv")

    # 保存验证实例结果
    if valid_results_100:
        column_names = [f"instance_{i}" for i in range(len(valid_results_100[0]))]
        results_100_data = []
        for epoch_results in valid_results_100:
            if hasattr(epoch_results, 'to'):
                epoch_results = epoch_results.to('cpu').numpy()
            results_100_data.append(epoch_results.tolist())

        df_100 = pd.DataFrame(results_100_data, columns=column_names)
        df_100.to_csv(os.path.join(save_manager.experiment_dir, "results", "validation_instances.csv"), index=False)

    # 输出实验总结
    experiment_summary = save_manager.get_experiment_summary()
    print("\n" + "="*60)
    print("实验总结:")
    print(f"实验目录: {experiment_summary['experiment_dir']}")
    print(f"最佳makespan: {experiment_summary['best_models']['makespan']['metric_value']:.4f}")
    print(f"总训练时间: {training_log['total_time']:.2f}秒")
    print(f"总迭代次数: {training_log['total_iterations']}")
    print("="*60)

    print("total_time: ", time.time() - start_time)


if __name__ == '__main__':
    main()
