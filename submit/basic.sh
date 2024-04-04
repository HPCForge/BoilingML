#!/bin/bash
#SBATCH -p free-gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:A30:2
#SBATCH --mem=32G
#SBATCH --time=016:00:00
#SBATCH --mail-user=nsankar1@uci.edu
#SBATCH --mail-type=END

module load python/3.10.2
module load cuda/11.7.1
module load gcc/11.2.0

source /pub/nsankar1/envs/sciml/bin/activate

python sciml/train.py

