try:
    from gymnasium.envs.registration import register
    # Registrar for the gymnasium environment
    register(
        id='fjsp-v0',  # Environment name (including version number)
        entry_point='env.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
        disable_env_checker=True,
    )
    # 注册USV（无人船）环境 - USV训练与验证模块实施计划第一阶段
    register(
        id='usv-v0',  # USV环境名称
        entry_point='env.usv_env:USVEnv',  # USV环境类位置
        disable_env_checker=True,
    )
except ImportError:
    # 如果gymnasium不可用，尝试使用gym（向后兼容）
    try:
        from gym.envs.registration import register
        register(
            id='fjsp-v0',  # Environment name (including version number)
            entry_point='env.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
            disable_env_checker=True,
        )
        # 注册USV（无人船）环境 - USV训练与验证模块实施计划第一阶段
        register(
            id='usv-v0',  # USV环境名称
            entry_point='env.usv_env:USVEnv',  # USV环境类位置
            disable_env_checker=True,
        )
    except ImportError:
        # 如果两者都不可用，则跳过注册
        pass
