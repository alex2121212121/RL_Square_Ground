import os
import socket
import numpy as np
import csv
import sys
import os

from Env2DCylinderModified import Env2DCylinderModified
from probe_positions import probe_positions
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
import numpy as np
from dolfin import Expression
#from printind.printind_function import printi, printiv
import math
from stable_baselines3.common.monitor import Monitor
from gym.wrappers.time_limit import TimeLimit
import os

import argparse
import os
import json
#import pandas as pd
#from tqdm import trange
from sb3_contrib import TQC
from stable_baselines3 import SAC
from Env2DCylinderModified import Env2DCylinderModified
from simulation_base.env import resume_env, nb_actuations, simulation_duration
from stable_baselines3.common.evaluation import evaluate_policy

# If previous evaluation results exist, delete them
if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")


if __name__ == '__main__':

    ## Things to modify before single run (evaluate the policy)
    saver_restore ='/rds/general/user/ad6318/home/rwd_test2_RL_2DCylinder_FlowControl_SB3/Cylinder2DFlowControlWithRL/saver_data/TQC35FStraineval_model_351000_steps.zip'
    vecnorm_path = '/rds/general/user/ad6318/home/rwd_test2_RL_2DCylinder_FlowControl_SB3/Cylinder2DFlowControlWithRL/saver_data/TQC35FStraineval_model_vecnormalize_351000_steps.pkl'
    #saver_restore ='/rds/general/user/cx220/home/TQCPM0FS/RL_UROP-master/Cylinder2DFlowControlWithRL/saver_data/TQCPM0FS_model_869375_steps.zip'
    #vecnorm_path = '/rds/general/user/cx220/home/TQCPM0FS/RL_UROP-master/Cylinder2DFlowControlWithRL_TQC35FS_savenormalize/saver_data/TQC35FStraineval_model_vecnormalize_520000_steps.pkl'

    #'/rds/general/user/jz1720/home/TQCPM0FS/RL_UROP-master/Cylinder2DFlowControlWithRL/saver_data/TQC4SP30FS_model_620425_steps.zip'
    
    horizon = 800 # Number of actions for single run. Non-dimensional time is horizon*action_step_size (usually horizon*0.5)
    
    agent = TQC.load(saver_restore)
    env = SubprocVecEnv([resume_env(plot=False, dump_vtu=200, single_run=True, horizon=horizon, n_env=99)], start_method='spawn')
    env = VecFrameStack(env, n_stack=35)
    env = VecNormalize.load(venv=env, load_path=vecnorm_path)

    observations = env.reset()
    #example_environment.render = True

    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    #single_run_duration = horizon*action_step_size  # In non-dimensional time
    action_steps = int(horizon)
    #evaluate_policy(model, env, n_eval_episodes=1,deterministic=True)
    #internals = agent.initial_internals()
    
    episode_reward = 0.0
    for k in range(action_steps):
        action, _ = agent.predict(observations, deterministic=True)
        observations, rw, done, _ = env.step(action)
        episode_reward += rw
        print("Reward:", episode_reward)