# ============================================================================
# main.py - USV调度系统训练与评估主程序 (含甘特图绘制)
# ============================================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 用于图例
import time
import pickle
import glob

# --- 修复 OMP Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------

from instance_generator import InstanceConfig
from env import USVSchedulingEnv
from ppo_test import PPO

# 创建保存目录
os.makedirs('./results', exist_ok=True)
os.makedirs('./models', exist_ok=True)


class InstanceLoader:
    """算例加载器 (保持不变)"""

    def __init__(self, data_dir, n_usvs, n_tasks, train_ratio=0.8):
        self.pattern = os.path.join(data_dir, f"u{n_usvs}_t{n_tasks}_*.pkl")
        self.files = sorted(glob.glob(self.pattern))

        if not self.files:
            raise ValueError(f"No instance files found matching: {self.pattern}")

        split_idx = int(len(self.files) * train_ratio)
        self.train_files = self.files[:split_idx]
        self.test_files = self.files[split_idx:]

        print(f"Loaded {len(self.files)} instances for U{n_usvs}-T{n_tasks}")

    def get_train_instance(self):
        fpath = np.random.choice(self.train_files)
        with open(fpath, 'rb') as f: return pickle.load(f)

    def get_test_instances(self):
        instances = []
        for fpath in self.test_files:
            with open(fpath, 'rb') as f: instances.append(pickle.load(f))
        return instances


def evaluate(ppo_agent, test_instances):
    """评估函数 (保持不变)"""
    total_rewards = []
    makespans = []

    for instance in test_instances:
        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            real_action, _, _, _, _ = ppo_agent.select_action(env, state, deterministic=True)
            if real_action is None: break
            state, reward, done, info = env.step(real_action)
            ep_reward += reward
            steps += 1
            if steps > env.n_tasks * 3: break

        total_rewards.append(ep_reward)
        makespan = info.get('makespan', env.current_time) if done else 5000.0
        makespans.append(makespan)

    return np.mean(total_rewards), np.mean(makespans)


def plot_results(rewards, makespans, losses):
    """绘制训练曲线 (保持不变)"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1);
    plt.plot(rewards);
    plt.title("Reward");
    plt.grid(True)
    plt.subplot(1, 3, 2);
    plt.plot(makespans);
    plt.title("Test Makespan");
    plt.grid(True)
    plt.subplot(1, 3, 3);
    plt.plot(losses);
    plt.title("Loss");
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./results/training_curves.png')
    plt.close()


def plot_gantt_chart(env, save_path='./results/schedule_gantt.png'):
    """
    根据环境详细历史记录绘制甘特图
    包含：任务执行、移动过程、充电过程
    """
    print(f"正在生成详细甘特图 -> {save_path}")

    n_usvs = env.n_usvs
    history = env.usv_history

    fig, ax = plt.subplots(figsize=(14, 8))

    # 颜色配置
    # 任务使用 tab20 颜色板
    task_cmap = plt.get_cmap('tab20')
    # 移动使用浅灰色
    color_move = '#D3D3D3'
    # 充电使用黄色/橙色
    color_charge = '#FFD700'

    # 记录最大时间用于设置坐标轴
    max_time = 0

    # 遍历每个 USV 的历史记录
    for usv_id in range(n_usvs):
        events = history[usv_id]

        for event in events:
            etype = event['type']
            start = event['start']
            end = event['end']
            duration = end - start
            info = event.get('info', '')

            if end > max_time:
                max_time = end

            if duration <= 0.01: continue  # 跳过极短的时间片

            if etype == 'task':
                # 任务块
                # 尝试解析任务ID来分配固定颜色
                try:
                    tid = int(info.replace('T', ''))
                    color = task_cmap(tid % 20)
                except:
                    color = 'skyblue'

                rect = ax.barh(y=usv_id, width=duration, left=start,
                               height=0.6, align='center',
                               color=color, edgecolor='black', alpha=0.9)

                # 文字标签
                ax.text(start + duration / 2, usv_id, info,
                        ha='center', va='center', color='white',
                        fontsize=8, fontweight='bold')

            elif etype == 'move':
                # 移动块 (稍微窄一点，以示区别)
                rect = ax.barh(y=usv_id, width=duration, left=start,
                               height=0.4, align='center',
                               color=color_move, edgecolor='gray', alpha=0.6, hatch='///')

            elif etype == 'charge':
                # 充电块
                rect = ax.barh(y=usv_id, width=duration, left=start,
                               height=0.6, align='center',
                               color=color_charge, edgecolor='black', alpha=1.0)

                ax.text(start + duration / 2, usv_id, "⚡",
                        ha='center', va='center', color='black', fontsize=10)

    # --- 设置图表格式 ---
    ax.set_yticks(range(n_usvs))
    ax.set_yticklabels([f'USV-{i}' for i in range(n_usvs)], fontsize=11, fontweight='bold')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('USV Detailed Schedule (Task / Move / Charge)', fontsize=15)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5, which='both')
    ax.minorticks_on()

    ax.set_xlim(0, max_time * 1.05)

    # --- 创建自定义图例 ---
    legend_patches = [
        mpatches.Patch(facecolor=task_cmap(0), edgecolor='black', label='Task Execution'),
        mpatches.Patch(facecolor=color_move, edgecolor='gray', hatch='///', alpha=0.6, label='Moving'),
        mpatches.Patch(facecolor=color_charge, edgecolor='black', label='Charging')
    ]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=3)

    plt.tight_layout()
    # 增加底部边距以放置图例
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print("详细甘特图生成完毕。")


def main():
    # -----------------------------------------
    # 1. 配置
    # -----------------------------------------
    N_USVS = 4
    N_TASKS = 20
    DATA_DIR = './data/train_eval'

    MAX_EPOCHS = 100
    EVAL_INTERVAL = 10
    UPDATE_TIMESTEP = 2000
    LR = 3e-4

    # -----------------------------------------
    # 2. 数据加载与初始化
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        loader = InstanceLoader(DATA_DIR, N_USVS, N_TASKS)
    except ValueError as e:
        print(e)
        return

    ppo = PPO(hidden_dim=64, hgnn_layers=2, lr=LR)

    history = {'rewards': [], 'test_makespans': [], 'losses': []}
    best_makespan = float('inf')
    timestep_counter = 0
    test_set = loader.get_test_instances()

    print("Start Training...")
    start_time = time.time()

    # -----------------------------------------
    # 3. 训练循环
    # -----------------------------------------
    for i_epoch in range(1, MAX_EPOCHS + 1):
        instance = loader.get_train_instance()
        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            timestep_counter += 1
            real_action, action_idx, log_prob, value, avail_acts = ppo.select_action(env, state)

            if real_action is None: break

            next_state, reward, done, info = env.step(real_action)

            ppo.store_transition(state, avail_acts, action_idx, reward, log_prob, done, value)

            state = next_state
            ep_reward += reward

            if timestep_counter % UPDATE_TIMESTEP == 0:
                update_info = ppo.update(n_epochs=4, batch_size=64)
                if update_info: history['losses'].append(update_info['loss'])

        history['rewards'].append(ep_reward)

        # -----------------------------------------
        # 4. 评估
        # -----------------------------------------
        if i_epoch % 10 == 0:
            print(f"Epoch {i_epoch} | Reward: {ep_reward:.2f}")

        if i_epoch % EVAL_INTERVAL == 0:
            test_reward, test_makespan = evaluate(ppo, test_set)
            history['test_makespans'].append(test_makespan)

            print(f"--- Eval @ Epoch {i_epoch} ---")
            print(f"Test Makespan: {test_makespan:.2f}")

            if test_makespan < best_makespan:
                best_makespan = test_makespan
                ppo.save('./models/best_model.pth')

            plot_results(history['rewards'], history['test_makespans'], history['losses'])

    print(f"Training Finished! Best Makespan: {best_makespan:.2f}")

    # -----------------------------------------
    # 5. 最终演示与画图
    # -----------------------------------------
    print("\n运行最终演示并生成甘特图...")
    # 加载最佳模型
    ppo.policy.load_state_dict(torch.load('./models/best_model.pth'))

    # 挑选一个测试实例进行演示
    demo_instance = test_set[0]  # 或者使用 loader.get_train_instance()
    env = USVSchedulingEnv(demo_instance)
    state = env.reset()
    done = False

    print(f"演示实例: {env.n_usvs} USVs, {env.n_tasks} Tasks")

    while not done:
        real_action, _, _, _, _ = ppo.select_action(env, state, deterministic=True)
        if real_action is None:
            print("所有USV无法继续执行任务（可能电量耗尽）。")
            break

        # 打印一步调度信息
        usv_id = real_action % env.n_usvs
        task_id = real_action // env.n_usvs
        print(f"Assign Task {task_id} -> USV {usv_id}")

        state, _, done, info = env.step(real_action)

    print(f"Final Makespan: {info.get('makespan', 'N/A')}")

    # === 调用甘特图绘制函数 ===
    plot_gantt_chart(env, save_path='./results/final_schedule_gantt.png')


if __name__ == "__main__":
    main()