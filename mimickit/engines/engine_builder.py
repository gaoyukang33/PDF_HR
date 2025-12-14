try:
    import isaacgym.gymapi as gymapi
except:
    pass

def build_engine(config, num_envs, device, visualize):
    eng_name = config["engine_name"]

    if (eng_name == "isaac_gym"):
        import engines.isaac_gym_engine as isaac_gym_engine
        engine = isaac_gym_engine.IsaacGymEngine(config, num_envs, device, visualize)
    elif (eng_name == "isaac_lab"):
        import engines.isaac_lab_engine as isaac_lab_engine
        engine = isaac_lab_engine.IsaacLabEngine(config, num_envs, device, visualize)
    elif ('viser' in eng_name):
        config["engine_name"] = eng_name.replace('_viser', '')
        engine = build_engine(config, num_envs, device, False)
    else:
        assert False, print("Unsupported engine: {:s}".format(eng_name))

    return engine