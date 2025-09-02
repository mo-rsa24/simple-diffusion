#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL      # Notify on job completion and fail
#SBATCH --mail-user=iammrrobot24@gmail.com  # Set your email address
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclude=mscluster45

source ~/.bashrc
mamba activate superdiff

python3 main.py --config cifar/configs/sm/cifar/vpsde.py --workdir $PWD/cifar/chkpt/ --mode train
