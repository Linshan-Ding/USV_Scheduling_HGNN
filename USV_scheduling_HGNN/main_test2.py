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


# ============================================================================
# 新增：甘特图绘制函数
# ============================================================================
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


# ============================================================================
# 优化：科研风格路径规划图绘制函数 (含充电返航路径)
# ============================================================================
def plot_trajectory_map(env, save_path='./results/final_trajectory_map.png'):
    """
    绘制所有 USV 的移动路径图
    改进点：
    1. 包含任务点访问路径 (实线)
    2. 包含返回起始点充电/结束的路径 (虚线)
    3. 标注距离和关键节点
    """
    print(f"正在生成路径规划图 (含充电路线) -> {save_path}")

    n_usvs = env.n_usvs
    task_coords = env.task_coords
    task_states = env.task_states  # [scheduled, start, end, usv_id]
    history = env.usv_history

    # 获取 Depot 坐标 (假设在 env config 中，如果没有则默认为 (0,0))
    # 注意：之前的 env.py 中我们是在 reset 里将 usv_states 归零作为 depot，
    # 这里我们尝试从 config 获取，或者默认 (0,0)
    if hasattr(env, 'config') and hasattr(env.config, 'depot_position'):
        depot_pos = np.array(env.config.depot_position)  # 如果 config 是对象
    elif isinstance(env.config, dict) and 'depot_position' in env.config:
        depot_pos = np.array(env.config['depot_position'])  # 如果 config 是字典
    else:
        depot_pos = np.array([0.0, 0.0])  # 默认值

    # 设置绘图风格
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 12))

    # 颜色板
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_usvs)]

    # 1. 绘制底图：所有任务点
    ax.scatter(task_coords[:, 0], task_coords[:, 1], c='whitesmoke', s=150,
               edgecolors='lightgray', zorder=1)

    # 标注未分配的任务 (理论上应该没有，如果有则显示灰色)
    for i, (x, y) in enumerate(task_coords):
        if task_states[i, 0] == 0:
            ax.text(x, y, f"T{i}", fontsize=8, color='gray', ha='center', va='center')

    # 2. 绘制 Depot
    ax.scatter(depot_pos[0], depot_pos[1], c='gold', marker='p', s=500,
               edgecolors='black', linewidth=1.5, zorder=10, label='Depot (Charge)')
    ax.text(depot_pos[0], depot_pos[1] - 40, "DEPOT", fontweight='bold', ha='center')

    total_distance_fleet = 0
    legend_handles = []

    # 3. 遍历每个 USV 重建完整路径
    for usv_id in range(n_usvs):
        color = colors[usv_id % len(colors)]

        # --- A. 构建事件序列 ---
        # 我们需要按时间顺序排列所有要去的目标点
        # 事件类型: ('task', time, coords, label) 或 ('charge', time, coords, label)

        events = []

        # 1. 添加任务事件
        assigned_indices = np.where(task_states[:, 3] == usv_id)[0]
        for tid in assigned_indices:
            start_time = task_states[tid, 1]
            coords = task_coords[tid]
            events.append({
                'type': 'task',
                'time': start_time,
                'pos': coords,
                'label': f"T{tid}"
            })

        # 2. 添加充电事件 (从 history 中提取)
        # history 中 type='charge' 的记录表示正在充电，说明之前一次移动是回 Depot
        usv_log = history.get(usv_id, [])
        for log in usv_log:
            if log['type'] == 'charge':
                events.append({
                    'type': 'charge',
                    'time': log['start'],  # 充电开始时间即到达Depot时间
                    'pos': depot_pos,
                    'label': 'Charge'
                })

        # 按时间排序
        events.sort(key=lambda x: x['time'])

        # --- B. 生成路径点序列 ---
        # 初始位置在 Depot
        current_pos = depot_pos
        path_segments = []  # 存储线段 [(start_pos, end_pos, line_style)]

        usv_dist = 0

        for event in events:
            target_pos = event['pos']
            dist = np.linalg.norm(target_pos - current_pos)

            if dist > 1e-3:  # 如果有移动
                # 判断线型：如果是去充电(目标是Depot)，用虚线；去任务，用实线
                if event['type'] == 'charge':
                    style = '--'
                    alpha = 0.5
                else:
                    style = '-'
                    alpha = 0.8

                path_segments.append((current_pos, target_pos, style, alpha))
                usv_dist += dist

                # 绘制连线
                ax.plot([current_pos[0], target_pos[0]], [current_pos[1], target_pos[1]],
                        color=color, linestyle=style, linewidth=2, alpha=alpha, zorder=2)

                # 绘制箭头
                mid_x = (current_pos[0] + target_pos[0]) / 2
                mid_y = (current_pos[1] + target_pos[1]) / 2
                ax.annotate('', xy=target_pos, xytext=current_pos,
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=alpha))

                # 标注距离 (仅长距离标注)
                if dist > 50:
                    ax.text(mid_x, mid_y, f"{dist:.0f}", fontsize=7, color=color,
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

            # 绘制节点
            if event['type'] == 'task':
                ax.scatter(target_pos[0], target_pos[1], color=color, s=120,
                           edgecolors='black', zorder=3)
                ax.text(target_pos[0], target_pos[1], event['label'],
                        fontsize=7, color='white', ha='center', va='center', fontweight='bold')

            current_pos = target_pos  # 更新位置

        # --- C. 最终返航 ---
        # 所有任务和中间充电完成后，USV 必须返回 Depot
        dist_home = np.linalg.norm(current_pos - depot_pos)
        if dist_home > 1e-3:
            path_segments.append((current_pos, depot_pos, ':', 0.4))
            usv_dist += dist_home

            # 绘制最终返航线 (点划线)
            ax.plot([current_pos[0], depot_pos[0]], [current_pos[1], depot_pos[1]],
                    color=color, linestyle=':', linewidth=2, alpha=0.6, zorder=2)
            # 箭头
            ax.annotate('', xy=depot_pos, xytext=current_pos,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.4))

        total_distance_fleet += usv_dist

        # 添加图例句柄
        patch = mpatches.Patch(color=color, label=f'USV-{usv_id} (Dist: {usv_dist:.1f})')
        legend_handles.append(patch)

    # 4. 图表装饰
    # 标注线型含义
    line_legend = [
        plt.Line2D([0], [0], color='gray', linestyle='-', lw=2, label='Task Travel'),
        plt.Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Return for Charge'),
        plt.Line2D([0], [0], color='gray', linestyle=':', lw=2, label='Final Return')
    ]

    # 合并图例
    first_legend = ax.legend(handles=legend_handles, loc='upper left',
                             bbox_to_anchor=(1.02, 1), title="USV Statistics")
    ax.add_artist(first_legend)
    ax.legend(handles=line_legend, loc='upper left',
              bbox_to_anchor=(1.02, 0.6), title="Path Types")

    ax.set_title(
        f"USV Multi-Agent Trajectory Map (With Charging & Return)\nTotal Fleet Distance: {total_distance_fleet:.2f} m",
        fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("X Coordinate (m)", fontsize=12)
    ax.set_ylabel("Y Coordinate (m)", fontsize=12)

    # 保持坐标轴比例一致
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("路径规划图生成完毕。")


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
    print("\n运行最终演示并生成图表...")
    # 尝试加载模型，如果没训练直接加载可能会报错，请确保有 ./models/best_model.pth
    ppo.policy.load_state_dict(torch.load('./models/best_model.pth'))
    # 挑选一个测试实例进行演示
    demo_instance = test_set[0]
    env = USVSchedulingEnv(demo_instance)
    state = env.reset()
    done = False

    print(f"演示实例: {env.n_usvs} USVs, {env.n_tasks} Tasks")

    while not done:
        real_action, _, _, _, _ = ppo.select_action(env, state, deterministic=True)
        if real_action is None:
            print("Action is None, Stop.")
            break

        state, _, done, info = env.step(real_action)

    print(f"Final Makespan: {info.get('makespan', 'N/A')}")

    # === 调用绘图函数 ===
    # 1. 绘制甘特图
    plot_gantt_chart(env, save_path='./results/final_schedule_gantt.png')

    # 2. 绘制移动路径图 (新增)
    plot_trajectory_map(env, save_path='./results/final_trajectory_map.png')


if __name__ == "__main__":
    main()