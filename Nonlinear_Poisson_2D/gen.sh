#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1  # Number of GPUs
#SBATCH --mem=20000  # Requested Memory
#SBATCH -t 10:00:00  # Job time limit


cd /work/pi_wuzhexu_umass_edu/FPGDM

module load python/3.8.5
module load anaconda/2022.10

conda activate tg3

python gen_P_2d.py -ns 200 -nex 40 -nx 16 -m 8 -d0 -0.0003 -d1 1 -d2 1 -alp 1 -tau 7 -flg 1 -seed 9 
python gen_P_2d.py -ns 10 -nex 10 -nx 16 -m 8 -d0 -0.0003 -d1 1 -d2 1 -alp 1 -tau 7 -flg 2 -seed 25