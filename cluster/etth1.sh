#!/usr/bin/env bash
#SBATCH --job-name=informer_train
#SBATCH --output=informer_train%j.log
#SBATCH --error=informer_train%j.err
#SBATCH --mail-user=rosello@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
export PATH="/home/rosello/anaconda3/bin:$PATH"
source activate informer-env
srun -u ../scripts/ETTh1.sh ${1} ${2}