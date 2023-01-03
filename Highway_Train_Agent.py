import os
import time

import json

import random
import numpy as np
import torch

import gym

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import highway_env

import argparse


DEFAULT_DIR = "data-v1"
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda:0"
DEFAULT_N_STEPS = 1000000
DEFAULT_N_EVAL = 200
DEFAULT_EVAL_FREQ = 25000
DEFAULT_LAYER_SIZE = 256

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--layer-size", type=int, default=DEFAULT_LAYER_SIZE)
    args = parser.parse_args()


    seed = args.seed


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    sb3.common.utils.set_random_seed(seed,using_cuda=True)


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    
    
    #Data Directory
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
        

        
        
    #Environnement
    env_name="highway-v0"
    if not os.path.isdir("{}/{}".format(data_dir,env_name)):
        os.mkdir("{}/{}".format(data_dir,env_name)) 

    with open("config.json", 'r') as fp:
        highway_config = json.load(fp)


    env_fn = lambda : gym.make(env_name, config=highway_config)

    # Multi CPU Training Env
    train_env = SubprocVecEnv([env_fn for _ in range(4)])
    train_env.seed(seed)
    train_env.reset()
    
    # Single CPU Evaluation Env
    eval_env = env_fn()
    eval_env.seed(seed)
    obs = eval_env.reset()

    




    
    #Agent
    agent_name = "Agent_{}".format(seed)
    if not os.path.isdir("{}/{}/{}".format(data_dir,env_name,agent_name)):
        os.mkdir("{}/{}/{}".format(data_dir,env_name,agent_name))
        
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[args.layer_size, dict(vf=[args.layer_size], pi=[args.layer_size])])
    agent = PPO(MlpPolicy, train_env, batch_size=64, n_steps=512, n_epochs=10, learning_rate=2e-4,
                ent_coef=1e-3, gamma=0.95, gae_lambda=0.95, clip_range=0.02, target_kl=0.05, policy_kwargs=policy_kwargs,
                verbose=False, tensorboard_log="{}/{}/{}/tensorboard/".format(data_dir,env_name,agent_name), device=device, seed=seed)

    #Evaluation settings
    #    REAL_eval_freq  =  eval_freq x training_env_cpu_number
    evalcallback = EvalCallback(eval_env=eval_env, name="eval", n_eval_episodes=args.n_eval, eval_freq=args.eval_freq, deterministic=True, verbose=False)

    #Training
    agent.learn(total_timesteps=args.n_steps, reset_num_timesteps=False, callback=evalcallback)

    agent.save("{}/{}/{}/model".format(data_dir,env_name,agent_name))

