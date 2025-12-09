
import os
import sys
import json
import random
from typing import Dict, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入USV案例生成器
from usv_case_generator import (
    generate_all_cases,
    CASE_COMBINATIONS,
    USVCaseData
)


def usv_case_data_to_dict(case_data: USVCaseData) -> Dict:
    """
    将USVCaseData对象转换为可序列化的字典格式

    参数：
        case_data: USV案例数据对象

    返回：
        可序列化的字典
    """
    return {
        "基本信息": {
            "案例ID": case_data.case_id,
            "USV数量": case_data.num_usvs,
            "任务数量": case_data.num_tasks,
            "随机化程度": case_data.randomization_level,
            "种子": case_data.seed
        },
        "环境参数": {
            "地图尺寸": case_data.map_size,
            "起始点": case_data.start_point,
            "环境固定参数": case_data.environment_parameters
        },
        "USV数据": {
            "USV位置": case_data.usv_positions,
            "USV初始电量": case_data.usv_initial_energy
        },
        "任务数据": {
            "任务位置": case_data.task_positions,
            "任务类型": case_data.task_types,
            "任务执行时间期望值": case_data.task_execution_times,
            "任务模糊时间": case_data.task_fuzzy_times,
            "任务从起点航行时间": case_data.task_navigation_times
        },
        "邻接矩阵": {
            "任务-USV邻接矩阵": case_data.task_usv_adjacency,
            "任务前驱关系矩阵": case_data.task_predecessor
        }
    }


def generate_validation_data_for_combination(num_usvs: int, num_tasks: int,
                                          output_dir: str, num_instances: int = 10) -> None:
    """
    为特定任务-USV组合生成验证数据

    参数：
        num_usvs: USV数量
        num_tasks: 任务数量
        output_dir: 输出目录
        num_instances: 生成的实例数量
    """
    print(f"[INFO] 正在生成 {num_tasks}任务_{num_usvs}USV 的验证数据...")

    # 创建输出目录
    combination_dir = os.path.join(output_dir, f"{num_tasks}_{num_usvs}")
    os.makedirs(combination_dir, exist_ok=True)

    # 为该组合生成多个实例
    for instance_id in range(num_instances):
        # 使用不同的种子确保多样性
        seed = 10000 + num_tasks * 100 + num_usvs * 10 + instance_id
        random.seed(seed)

        try:
            from usv_case_generator import create_generator
            # 创建案例生成器
            generator = create_generator(
                num_usvs=num_usvs,
                num_tasks=num_tasks,
                path=output_dir,
                flag_doc=False,  # 内存模式
                randomization_level="medium",
                seed=seed
            )

            # 生成案例
            case_data = generator.get_case(idx=0)

            # 转换为可序列化格式
            case_dict = usv_case_data_to_dict(case_data)

            # 保存为JSON文件
            filename = f"usv_case_{num_tasks}_{num_usvs}_instance_{instance_id+1:02d}.json"
            filepath = os.path.join(combination_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(case_dict, f, indent=2, ensure_ascii=False)

            print(f"  [PASS] 已生成: {filename}")

        except Exception as e:
            print(f"  [ERROR] 生成失败: {str(e)}")
            continue


def main():
    """主函数：生成所有验证数据"""
    print("=" * 60)
    print("USV验证数据生成器")
    print("生成20种任务-USV组合的验证数据集")
    print("这些固定数据集将用于评估训练出的最佳模型性能")
    print("=" * 60)

    # 设置输出目录
    output_dir = "../usv_data_dev"
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    total_combinations = len(CASE_COMBINATIONS)
    successful_combinations = 0
    total_instances = 0

    print(f"[INFO] 开始生成 {total_combinations} 种组合的验证数据...")
    print(f"[INFO] 每种组合生成 10 个实例")
    print("-" * 60)

    # 为每种组合生成验证数据
    for idx, (num_usvs, num_tasks) in enumerate(CASE_COMBINATIONS, 1):
        print(f"[{idx:2d}/{total_combinations}] 处理组合: {num_tasks}任务 × {num_usvs}USV")

        try:
            generate_validation_data_for_combination(
                num_usvs=num_usvs,
                num_tasks=num_tasks,
                output_dir=output_dir,
                num_instances=10
            )
            successful_combinations += 1
            total_instances += 10

        except Exception as e:
            print(f"  [ERROR] 组合 {num_tasks}_{num_usvs} 生成失败: {str(e)}")
            continue

    # 输出统计结果
    print("-" * 60)
    print("[SUCCESS] 验证数据生成完成!")
    print(f"成功组合: {successful_combinations}/{total_combinations}")
    print(f"总实例数: {total_instances}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 验证目录结构
    print("\n[INFO] 验证生成的目录结构:")
    for usv_count in [2, 4, 6, 8]:
        for task_count in [40, 60, 80, 100, 120]:
            dir_path = os.path.join(output_dir, f"{task_count}_{usv_count}")
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
                print(f"  {task_count}_{usv_count}/: {len(files)} 个文件")


if __name__ == "__main__":
    main()