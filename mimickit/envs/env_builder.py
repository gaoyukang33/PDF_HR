import yaml
import envs.sim_env as sim_env

from util.logger import Logger

def build_env(env_file, num_envs, device, visualize):
    env_config = load_env_file(env_file)

    env_name = env_config["env_name"]
    Logger.print("Building {} env".format(env_name))

    if (env_name == "char"):
        import envs.char_env as char_env
        env = char_env.CharEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "deepmimic"):
        import envs.deepmimic_env as deepmimic_env
        env = deepmimic_env.DeepMimicEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "deepmimic_nrdf"):
        import envs.deepmimic_env_nrdf as deepmimic_env_nrdf
        env = deepmimic_env_nrdf.DeepMimicEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "amp"):
        import envs.amp_env as amp_env
        env = amp_env.AMPEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "amp_nrdf"):
        import envs.amp_env_nrdf as amp_env_nrdf
        env = amp_env_nrdf.AMPEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "ase"):
        import envs.ase_env as ase_env
        env = ase_env.ASEEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "add"):
        import envs.add_env as add_env
        env = add_env.ADDEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "add_nrdf"):
        import envs.add_env_nrdf as add_env_nrdf
        env = add_env_nrdf.ADDEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "char_dof_test"):
        import envs.char_dof_test_env as char_dof_test_env
        env = char_dof_test_env.CharDofTestEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "view_motion"):
        import envs.view_motion_env as view_motion_env
        env = view_motion_env.ViewMotionEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "task_location"):
        import envs.task_location_env as task_location_env
        env = task_location_env.TaskLocationEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "task_location_nrdf"):
        import envs.task_location_env_nrdf as task_location_env_nrdf
        env = task_location_env_nrdf.TaskLocationEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "task_steering"):
        import envs.task_steering_env as task_steering_env
        env = task_steering_env.TaskSteeringEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "task_steering_nrdf"):
        import envs.task_steering_env_nrdf as task_steering_env_nrdf
        env = task_steering_env_nrdf.TaskSteeringEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "static_objects"):
        import envs.static_objects_env as static_objects_env
        env = static_objects_env.StaticObjectsEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "static_objects_nrdf"):
        import envs.static_objects_env_nrdf as static_objects_env_nrdf
        env = static_objects_env_nrdf.StaticObjectsEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "static_objects_add"):
        import envs.static_objects_add_env as static_objects_add_env
        env = static_objects_add_env.StaticObjectsEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif (env_name == "static_objects_add_nrdf"):
        import envs.static_objects_add_env_nrdf as static_objects_add_env_nrdf
        env = static_objects_add_env_nrdf.StaticObjectsEnv(config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    elif ('record' in env_name):
        import envs.record_env as record_env
        env = record_env.RecordEnvBuilder(env_name, config=env_config, num_envs=num_envs, device=device, visualize=visualize)
    else:
        assert(False), "Unsupported env: {}".format(env_name)

    return env

def load_env_file(file):
    with open(file, "r") as stream:
        env_config = yaml.safe_load(stream)
    return env_config
