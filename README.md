# RL_2DCylinder_FlowControl_SB3

This repository contains code for training Reinforcement Learning based control aimed at reducing the drag due to vortex shedding in the wake of a rectangular cylinder in 2D.

The code is a further development (by Gonzalo Ancochea Blanco, Lucien Viala, Hakan Serpen, Chengwei Xia and Jacky Zhang, who are students supervised by Dr. Georgios Rigas at Imperial College London) on the work published in "Artificial Neural Networks trained through Deep Reinforcement Learning discover control strategies for active flow control", Rabault et. al., Journal of Fluid Mechanics (2019), preprint accessible at https://arxiv.org/pdf/1808.07664.pdf， and in "Accelerating Deep Reinforcement Learning strategies of Flow Control through a multi-environment approach", Rabault and Kuhnle, Physics of Fluids (2019), preprint accessible at https://arxiv.org/abs/1906.10382. Code available at https://github.com/jerabaul29/Cylinder2DFlowControlDRL and https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel respectively.

The Reinforcement Learning framework used in this code is based on Stable Baseline3 https://github.com/DLR-RM/stable-baselines3 and Stable Baseline3 Contrib https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

If you find this work useful and / or use it in your own research, please cite these works:

```
Rabault, J., Kuhnle, A (2019).
Accelerating Deep Reinforcement Leaning strategies of Flow Control through a
multi-environment approach.
Physics of Fluids.

Rabault, J., Kuchta, M., Jensen, A., Réglade, U., & Cerardi, N. (2019).
Artificial neural networks trained through deep reinforcement learning discover
control strategies for active flow control.
Journal of Fluid Mechanics, 865, 281-302. doi:10.1017/jfm.2019.62

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021).
Stable-Baselines3: Reliable Reinforcement Learning Implementations.
Journal of Machine Learning Research.
```

## Getting started

This code is mainly developed for the cluster at Imperial College London (HPC service) and can be adapted to run on PC (not recommended due to computation time) with minor modifications (PC version not developed yet).

The main code is located in **Cylinder2DFlowControlWithRL**. 
The simulation template to be run is in the **simulation_base** folder. If you want to run different simulations, the main file to modify is **env.py** with settings of simulations.

## Package installation on cluster
- Login to the cluster: ssh -XY ICusername@login.hpc.imperial.ac.uk
- For the first time, set up a personal python environment with:
```
module load anaconda3/personal
anaconda-setup
```
- Create an individual environment and activate it:
```
conda create -n RLSB3 -c conda-forge fenics (This will create an environment with a user-defined name "RLSB3" and install FEniCS with its dependencies.)
source activate RLSB3 (Conda deactivate if you want to quit the current environment)
```
- Install other packages required (SB3 needs python 3.7+ and PyTorch >= 1.11)
```
pip3 install torch torchvision torchaudio (Check https://pytorch.org/get-started/locally/ if it doesn't work)
pip install sb3-contrib (This will install stable-baselines3 as well automatically. Check https://sb3-contrib.readthedocs.io/en/master/guide/install.html for details.)
pip install tensorboard
(Other small packages like pickle5, scipy and peakutils can be also easily installed by pip)
```
- For details of all the packages and version, please refer to **RLSB3_requirements.txt**.

## Implementing on cluster as batch jobs 

The main script for launching trainings as batch jobs is the **script_launch_parallel_cluster.sh** script. This script specifies the settings of the job (Time, Number of Procs etc.) and calls **launch_parallel_training.py**, which actually setup and run the training process.

Make the job is sized correctly. For a mesh of around 10000 elements and a timestep of dt=0.004, these conservative guidelines are a good starting point:
- wall_time = 30 minutes * #_episodes / #_parallel environments
- n_cpus = #_parallel environments + 2

The job submission requires an environment variable **NUM_PORT** to be set prior to execution. This variable determines the number of parallel environments during training and can be modified in **script_launch_parallel_cluster.sh**.

## Updates based on the code from Jean Rabault et al.



## Troubleshooting

If you encounter problems, please:

- look for help in the .md readme files of this repo
- look for help on the github repo of the JFM paper used for serial training
- if this is not enough to get your problem solved, feel free to open an issue and ask for help.

## Main scripts

- **script_launch_parallel_cluster.sh**: automatically launch the training as a parallel batch job. This script calls **launch_parallel_training.py**.
- **launch_parallel_training.py**: define training parameters (Algorithm, Neural Network, hyperparameters etc.)
- **Run.sh**: launch a job to evaluate a particular policy. This script calls **single_runner.py**.
- **single_runner.py**: evaluate the latest saved policy.


## CFD simulation fenics, and user-defined user cases

The CFD simulation is implemented by FEniCS. Please refer to its official website https://fenicsproject.org/ for more details. 
For more details about the CFD simulation and how to build your own user-defined cases, please consult the Readme of the JFM code, availalbe at https://github.com/jerabaul29/Cylinder2DFlowControlDRL.

