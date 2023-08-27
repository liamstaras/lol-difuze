#!/bin/sh
#SBATCH -o slurm/%j.log
#SBATCH --job-name train-lol-512-difuze            # this is a parameter to help you sort your job when listing it
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --mem-per-cpu 10000
#SBATCH --partition shared-gpu,private-dpt-gpu
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --gpus ampere:1
#SBATCH --gres VramPerGpu:80GiB

source ./environment.sh
echo SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES
python lol-training-512.py "$@"
deactivate