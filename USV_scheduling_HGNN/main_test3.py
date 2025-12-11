# ============================================================================
# main.py - USV多智能体协同调度系统 (完整版)
# ============================================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import pickle
import glob
import re
import sys  # <--- 确保导入 sys
from collections import deque

# --- 修复 Windows 下 OpenMP 冲突错误 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 导入自定义模块 ---
# 请确保 env.py, ppo.py, hgnn.py, mlp.py 在同一目录下
from env import USVSchedulingEnv
from ppo import PPO
import instance_generator_mix # <--- 导入原模块

sys.modules['__main__'].InstanceConfig = instance_generator_mix.InstanceConfig
# 创建输出目录
os.makedirs('./results', exist_ok=True)
os.makedirs('./models', exist_ok=True)


# ============================================================================
# 1. 混合算例加载器 (Mixed Instance Loader)
# ============================================================================
class MixedInstanceLoader:
    """
    负责加载不同规模的 (USV, Task) 算例文件
    """

    def __init__(self, data_dir, train_ratio=0.8):
        self.data_dir = data_dir
        self.datasets = {}
        self.all_configs = []

        # 扫描文件
        pattern_str = os.path.join(data_dir, "*.pkl")
        all_files = glob.glob(pattern_str)

        if not all_files:
            raise ValueError(f"错误: 在 {data_dir} 未找到 .pkl 文件。请先运行 instance_generator.py 生成数据。")

        print(f"共扫描到 {len(all_files)} 个算例文件。")

        # 正则解析文件名: u4_t20_0.pkl
        # 假设生成器命名格式为: u{n_usvs}_t{n_tasks}_{index}.pkl
        filename_pattern = re.compile(r"u(\d+)_t(\d+)_(\d+).pkl")

        for fpath in all_files:
            fname = os.path.basename(fpath)
            match = filename_pattern.match(fname)
            if match:
                u = int(match.group(1))
                t = int(match.group(2))

                key = (u, t)
                if key not in self.datasets:
                    self.datasets[key] = []
                    self.all_configs.append(key)

                self.datasets[key].append(fpath)

        # 划分训练集和测试集
        self.train_pool = {}
        self.test_pool = {}

        for key in self.all_configs:
            files = sorted(self.datasets[key])
            split_idx = int(len(files) * train_ratio)
            # 保证至少有一个测试文件
            split_idx = min(split_idx, len(files) - 1)

            self.train_pool[key] = files[:split_idx]
            self.test_pool[key] = files[split_idx:]

        print(f"已加载 {len(self.all_configs)} 种配置: {sorted(self.all_configs)}")

    def get_random_train_instance(self):
        """训练时：随机选择一种配置，再从中随机选一个文件"""
        # 1. 随机选配置 (均匀采样，保证大小算例都能练到)
        config_idx = np.random.randint(len(self.all_configs))
        selected_config = self.all_configs[config_idx]

        # 2. 随机选文件
        file_list = self.train_pool[selected_config]
        fpath = np.random.choice(file_list)

        with open(fpath, 'rb') as f:
            return pickle.load(f), selected_config

    def get_fixed_test_instance(self):
        """测试时：固定选择最大规模算例 (10, 100) 进行高压测试"""
        target_config = (10, 100)

        # 如果还没生成 (10, 100)，则选最大的那个
        if target_config not in self.test_pool:
            # 按 Task 数量排序，取最大的
            target_config = sorted(self.all_configs, key=lambda x: x[1])[-1]

        # 固定取测试列表的第一个文件
        fpath = self.test_pool[target_config][0]

        with open(fpath, 'rb') as f:
            return pickle.load(f), target_config


# ============================================================================
# 2. 绘图功能 (科研风格)
# ============================================================================

def plot_gantt_chart(env, save_path='./results/final_schedule_gantt.png'):
    """绘制详细甘特图 (任务/移动/充电)"""
    print(f"生成甘特图 -> {save_path}")
    n_usvs = env.n_usvs
    history = env.usv_history

    fig, ax = plt.subplots(figsize=(14, 8))
    task_cmap = plt.get_cmap('tab20')
    color_move = '#D3D3D3'
    color_charge = '#FFD700'
    max_time = 0

    for usv_id in range(n_usvs):
        events = history[usv_id]
        for event in events:
            etype = event['type']
            start = event['start']
            end = event['end']
            duration = end - start
            info = event.get('info', '')

            if end > max_time: max_time = end
            if duration <= 0.01: continue

            if etype == 'task':
                try:
                    tid = int(info.replace('T', ''))
                    color = task_cmap(tid % 20)
                except:
                    color = 'skyblue'

                ax.barh(y=usv_id, width=duration, left=start, height=0.6, align='center',
                        color=color, edgecolor='black', alpha=0.9)
                ax.text(start + duration / 2, usv_id, info, ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold')

            elif etype == 'move':
                ax.barh(y=usv_id, width=duration, left=start, height=0.4, align='center',
                        color=color_move, edgecolor='gray', alpha=0.6, hatch='///')

            elif etype == 'charge':
                ax.barh(y=usv_id, width=duration, left=start, height=0.6, align='center',
                        color=color_charge, edgecolor='black', alpha=1.0)
                ax.text(start + duration / 2, usv_id, "⚡", ha='center', va='center', color='black')

    ax.set_yticks(range(n_usvs))
    ax.set_yticklabels([f'USV-{i}' for i in range(n_usvs)], fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(f'USV Schedule Gantt Chart (Makespan: {max_time:.2f}s)', fontsize=15)
    ax.set_xlim(0, max_time * 1.05)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    legend_patches = [
        mpatches.Patch(facecolor=task_cmap(0), edgecolor='black', label='Task Execution'),
        mpatches.Patch(facecolor=color_move, edgecolor='gray', hatch='///', alpha=0.6, label='Moving'),
        mpatches.Patch(facecolor=color_charge, edgecolor='black', label='Charging')
    ]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_trajectory_map(env, save_path='./results/final_trajectory_map.png'):
    """绘制路径规划图 (含充电返航路线)"""
    print(f"生成路径图 -> {save_path}")
    n_usvs = env.n_usvs
    task_coords = env.task_coords
    task_states = env.task_states
    history = env.usv_history

    # 假设 Depot 坐标 (如果 config 中没有，默认为 0,0)
    if hasattr(env, 'config') and isinstance(env.config, dict):
        # 这里的 env.config 是 dict (从 pickle 加载)
        # 注意：InstanceConfig dataclass 序列化后可能变成对象或 dict，视 pickle 方式而定
        # 这里做一个兼容处理
        depot_pos = np.array([0.0, 0.0])  # 默认
    else:
        depot_pos = np.array([0.0, 0.0])

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_usvs)]

    # 1. 绘制任务点
    ax.scatter(task_coords[:, 0], task_coords[:, 1], c='whitesmoke', s=150, edgecolors='lightgray', zorder=1)
    for i, (x, y) in enumerate(task_coords):
        ax.text(x, y, f"T{i}", fontsize=8, color='gray', ha='center', va='center')

    # 2. 绘制 Depot
    ax.scatter(depot_pos[0], depot_pos[1], c='gold', marker='p', s=400, edgecolors='black', zorder=10, label='Depot')

    total_dist = 0
    legend_handles = []

    # 3. 重建路径
    for usv_id in range(n_usvs):
        color = colors[usv_id]
        events = []

        # 提取任务事件
        assigned_idxs = np.where(task_states[:, 3] == usv_id)[0]
        for tid in assigned_idxs:
            events.append({'type': 'task', 'time': task_states[tid, 1], 'pos': task_coords[tid], 'label': f"T{tid}"})

        # 提取充电事件
        for log in history[usv_id]:
            if log['type'] == 'charge':
                events.append({'type': 'charge', 'time': log['start'], 'pos': depot_pos, 'label': 'Charge'})

        events.sort(key=lambda x: x['time'])

        current_pos = depot_pos
        usv_dist = 0

        for event in events:
            target_pos = event['pos']
            dist = np.linalg.norm(target_pos - current_pos)

            if dist > 1e-3:
                style = '--' if event['type'] == 'charge' else '-'
                alpha = 0.5 if event['type'] == 'charge' else 0.8

                ax.plot([current_pos[0], target_pos[0]], [current_pos[1], target_pos[1]],
                        color=color, linestyle=style, linewidth=2, alpha=alpha, zorder=2)

                # 箭头
                mid = (current_pos + target_pos) / 2
                ax.annotate('', xy=target_pos, xytext=current_pos,
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

                usv_dist += dist

            if event['type'] == 'task':
                ax.scatter(target_pos[0], target_pos[1], color=color, s=100, edgecolors='black', zorder=3)

            current_pos = target_pos

        # 最终返航
        dist_home = np.linalg.norm(current_pos - depot_pos)
        if dist_home > 1e-3:
            ax.plot([current_pos[0], depot_pos[0]], [current_pos[1], depot_pos[1]],
                    color=color, linestyle=':', linewidth=2, alpha=0.5, zorder=2)
            usv_dist += dist_home

        total_dist += usv_dist
        legend_handles.append(mpatches.Patch(color=color, label=f'USV-{usv_id} ({usv_dist:.0f}m)'))

    # 图表设置
    ax.set_title(f"Trajectory Map (Total Dist: {total_dist:.1f}m)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1), title="USV Stats")
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(rewards, makespans, losses):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Avg Reward")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(makespans)
    plt.title("Test Makespan")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title("PPO Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./results/training_curves.png')
    plt.close()


# ============================================================================
# 3. 主程序
# ============================================================================

def main():
    # --- 配置 ---
    DATA_DIR = './data/train_eval_mixed'
    MAX_EPOCHS = 2000  # 训练轮数
    EVAL_INTERVAL = 50  # 评估频率
    UPDATE_TIMESTEP = 2048  # PPO更新步数
    LR = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据加载
    try:
        loader = MixedInstanceLoader(DATA_DIR)
    except Exception as e:
        print(f"Error: {e}")
        print("请先运行 instance_generator.py 生成数据！")
        return

    # 2. 初始化 Agent
    # HGNN 支持变长输入，hidden_dim 设大一点以适应大规模算例
    ppo = PPO(hidden_dim=128, hgnn_layers=3, lr=LR)

    history = {'rewards': [], 'test_makespans': [], 'losses': []}
    best_makespan = float('inf')
    timestep_counter = 0

    print("开始混合规模训练...")

    # 3. 训练循环
    for i_epoch in range(1, MAX_EPOCHS + 1):
        # 随机获取一个算例 (可能是 4USV-20Task，也可能是 10USV-100Task)
        instance, config_info = loader.get_random_train_instance()

        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            timestep_counter += 1

            # 选择动作
            real_action, action_idx, log_prob, value, avail_acts = ppo.select_action(env, state)

            if real_action is None:
                break  # 无解或结束

            next_state, reward, done, info = env.step(real_action)

            # 存入 Buffer
            ppo.store_transition(state, avail_acts, action_idx, reward, log_prob, done, value)

            state = next_state
            ep_reward += reward

            # PPO 更新
            if timestep_counter % UPDATE_TIMESTEP == 0:
                update_info = ppo.update(n_epochs=4, batch_size=128)
                if update_info:
                    history['losses'].append(update_info['loss'])

        history['rewards'].append(ep_reward)

        # 打印日志
        if i_epoch % 10 == 0:
            print(f"Epoch {i_epoch} [{config_info}] | Reward: {ep_reward:.2f}")

        # 4. 评估 (固定使用最大规模算例)
        if i_epoch % EVAL_INTERVAL == 0:
            test_inst, test_conf = loader.get_fixed_test_instance()
            test_env = USVSchedulingEnv(test_inst)
            t_state = test_env.reset()
            t_done = False

            while not t_done:
                act, _, _, _, _ = ppo.select_action(test_env, t_state, deterministic=True)
                if act is None: break
                t_state, _, t_done, t_info = test_env.step(act)

            curr_makespan = t_info.get('makespan', 5000)
            history['test_makespans'].append(curr_makespan)

            print(f">>> Eval @ Ep {i_epoch} on {test_conf}: Makespan = {curr_makespan:.2f}")

            # 保存最佳模型
            if curr_makespan < best_makespan:
                best_makespan = curr_makespan
                ppo.save('./models/best_model.pth')
                print(">>> New Best Model Saved!")

            plot_training_curves(history['rewards'], history['test_makespans'], history['losses'])

    print("训练结束！")

    # --------------------------------------------------------
    # 5. 最终演示 (使用最佳模型 + 最大规模算例)
    # --------------------------------------------------------
    print("\n生成最终演示结果...")
    try:
        ppo.policy.load_state_dict(torch.load('./models/best_model.pth'))
    except:
        print("未找到最佳模型，使用最后一次的模型")

    demo_inst, demo_conf = loader.get_fixed_test_instance()
    env = USVSchedulingEnv(demo_inst)
    state = env.reset()
    done = False

    print(f"演示算例: {demo_conf}")

    while not done:
        act, _, _, _, _ = ppo.select_action(env, state, deterministic=True)
        if act is None: break
        state, _, done, info = env.step(act)

    print(f"最终 Makespan: {info.get('makespan')}")

    # 生成图表
    plot_gantt_chart(env)
    plot_trajectory_map(env)
    print("结果图已保存至 ./results/")


if __name__ == "__main__":
    main()