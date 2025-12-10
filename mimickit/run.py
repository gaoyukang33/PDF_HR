import numpy as np
import os
import shutil
import sys
import time

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize):
    env_file = args.parse_string("env_config")
    env = env_builder.build_env(env_file, num_envs, device, visualize)
    return env

def build_agent(agent_file, env, device):
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, out_model_file, int_output_dir, logger_type, log_file):
    agent.train_model(max_samples=max_samples, out_model_file=out_model_file, 
                      int_output_dir=int_output_dir, logger_type=logger_type,
                      log_file=log_file)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))
    return result

def create_output_dirs(out_model_file, int_output_dir):
    if (mp_util.is_root_proc()):
        output_dir = os.path.dirname(out_model_file)
        if (output_dir != "" and (not os.path.exists(output_dir))):
            os.makedirs(output_dir, exist_ok=True)
        
        if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
            os.makedirs(int_output_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)
        
    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def run(rank, num_procs, device, master_port, args):
    mode = args.parse_string("mode", "train")
    num_envs = args.parse_int("num_envs", 1)
    visualize = args.parse_bool("visualize", True)
    logger_type = args.parse_string("logger", "tb")
    log_file = args.parse_string("log_file", "output/log.txt")
    out_model_file = args.parse_string("out_model_file", "output/model.pt")
    int_output_dir = args.parse_string("int_output_dir", "")
    model_file = args.parse_string("model_file", "")

    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()

    create_output_dirs(out_model_file, int_output_dir)

    env = build_env(args, num_envs, device, visualize)
    
    out_model_dir = os.path.dirname(out_model_file)
    agent_file = args.parse_string("agent_config")
    agent = build_agent(agent_file, env, device)

    if (model_file != ""):
        agent.load(model_file)

    if (mode == "train"):
        env_file = args.parse_string("env_config")
        copy_file_to_dir(env_file, "env_config.yaml", out_model_dir)

        if (agent_file != ""):
            copy_file_to_dir(agent_file, "agent_config.yaml", out_model_dir)

        max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)
        train(agent=agent, max_samples=max_samples, out_model_file=out_model_file, 
              int_output_dir=int_output_dir, logger_type=logger_type, log_file=log_file)
    elif (mode == "test"):
        test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
        test(agent=agent, test_episodes=test_episodes)
    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def main(argv):
    root_rank = 0
    args = load_args(argv)
    master_port = args.parse_int("master_port", None)
    num_workers = args.parse_int("num_workers", 1)
    device = args.parse_string("device", "cuda:0")
    assert(num_workers > 0)
    
    # if master port is not specified, then pick a random one
    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for i in range(num_workers - 1):
        rank = i + 1
        if ("cuda" in device):
            curr_device = "cuda:" + str(rank)
        else:
            curr_device = device

        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)

    
    if (num_workers > 1 and "cuda" in device):
        curr_device = "cuda:" + str(root_rank)
    else:
        curr_device = device
    
    run(root_rank, num_workers, curr_device, master_port, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)
