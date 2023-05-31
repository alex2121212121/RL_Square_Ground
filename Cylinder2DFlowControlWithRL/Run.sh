#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=40gb
cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate RLSB3
python3 $PBS_O_WORKDIR/single_runner.py
