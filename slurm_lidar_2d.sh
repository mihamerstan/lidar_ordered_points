#!/bin/bash
#
#SBATCH --job-name=conv2d_32_32
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4,p40_4,v100_sxm2_4,v100_pci_2
#SBATCH --output=hpc_logs/slurmout_%A_%a.out
#SBATCH --error=hpc_logs/slurmout_%A_%a.err

python lidar_conv2d_train.py --num-epochs 300 --data-path ../lidar_data/32_32/ --batch_norm True
