import argparse
import os
import sys
import csv
import socket
import numpy as np
#from tqdm import tqdm
from simulation_base.env import resume_env, nb_actuations
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
from stable_baselines3.sac import SAC
import torch
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import CheckpointCallback
#from tensorforce.agents import Agent
#from tensorforce.execution import Runner


#from RemoteEnvironmentClient import RemoteEnvironmentClient


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    ap.add_argument("-s", "--savedir", required=False,
                    help="Directory into which to save the NN. Defaults to 'saver_data'.", type=str,
                    default='saver_data')

    args = vars(ap.parse_args())

    number_servers = args["number_servers"]
    savedir = args["savedir"]


    config = {}

    config["learning_rate"] = 1e-4
    config["learning_starts"] = 0
    config["batch_size"] = 128

    config["tau"] = 5e-3
    config["gamma"] = 0.99
    config["train_freq"] = 1
    config["target_update_interval"] = 1
    config["gradient_steps"] = 48

    config["buffer_size"] = int(10e5)
    config["optimize_memory_usage"] = False

    config["ent_coef"] = "auto_0.01"
    config["target_entropy"] = "auto"
    policy_kwargs = dict(net_arch=dict(pi=[512,512,512], qf=[512,512,512]))
    checkpoint_callback = CheckpointCallback(
                                            save_freq=max(10, 1),
                                            #num_to_keep=5,
                                            #save_buffer=True,
                                            #save_env_stats=True,
                                            save_path=savedir,
                                            name_prefix='TQC35FStraineval_model')

    #saver_restore ='/rds/general/user/cx220/home/TQCPM0FS/RL_UROP-master/Cylinder2DFlowControlWithRL_TQC35FS_tri512NN/saver_data/TQC35FStri512NN_model_100100_steps.zip'
    saver_restore ='/rds/general/user/ad6318/home/test_RL_2DCylinder_FlowControl_SB3/Cylinder2DFlowControlWithRL/saver_data/TQC35FStraineval_model_260000_steps.zip'
    env = SubprocVecEnv([resume_env(plot=False, dump_CL=False, dump_debug=10, n_env=i) for i in range(number_servers)], start_method='spawn')
    env = VecFrameStack(env, n_stack=35)
    env = VecNormalize(env, gamma=0.99)
    model = TQC.load(saver_restore, env=env)
    #model = TQC('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=savedir, **config)
    model.learn(15000000, callback=[checkpoint_callback], log_interval=1)
    #model.learn(15000000, log_interval=1)
   

    # name = "returns_tf.csv"
    # if (not os.path.exists("saved_models")):
    #     os.mkdir("saved_models")

    # # If continuing previous training - append returns
    # if (os.path.exists("saved_models/" + name)):
    #     prev_eps = np.genfromtxt("saved_models/" + name, delimiter=';', skip_header=1)
    #     offset = int(prev_eps[-1, 0])
    #     print(offset)
    #     with open("saved_models/" + name, "a") as csv_file:
    #         spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
    #         for ep in range(len(runner.episode_rewards)):
    #             spam_writer.writerow([offset + ep + 1, runner.episode_rewards[ep]])
    # # If strating training from zero - write returns
    # elif (not os.path.exists("saved_models/" + name)):
    #     with open("saved_models/" + name, "w") as csv_file:
    #         spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
    #         spam_writer.writerow(["Episode", "Return"])
    #         for ep in range(len(runner.episode_rewards)):
    #             spam_writer.writerow([ep + 1, runner.episode_rewards[ep]])





    print("Agent and Runner closed -- Learning complete -- End of script")
    os._exit(0)

