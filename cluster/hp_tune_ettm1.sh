#!/usr/bin/env bash
#SBATCH --job-name=informer_train
#SBATCH --output=informer_train%j.log
#SBATCH --error=informer_train%j.err
#SBATCH --mail-user=rosello@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
export PATH="/home/praphul/anaconda3/bin:$PATH"
source activate informer-env

srun -u python  hp_search.py --model informer   --data ETTm1  --attn qs --des 'Exp' --itr 5 --factor 5 --batch_size 16 --train_epochs 12
srun -u python  hp_search.py --model informer   --data ETTm1  --attn lam --des 'Exp' --itr 5 --factor 5 --batch_size 16 --train_epochs 12
