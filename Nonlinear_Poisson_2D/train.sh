#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1  # Number of GPUs
#SBATCH --mem=100000  # Requested Memory
#SBATCH -t 10:00:00  # Job time limit


cd /work/pi_wuzhexu_umass_edu/FPGDM

module load python/3.8.5
module load anaconda/2022.10

conda activate tg3
 
python main_2D.py