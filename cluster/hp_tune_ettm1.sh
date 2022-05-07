#!/usr/bin/env bash
#SBATCH --job-name=informer_train
#SBATCH --output=informer_train%j.log
#SBATCH --error=informer_train%j.err
#SBATCH --mail-user=praphul@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
export PATH="/home/praphul/anaconda3/bin:$PATH"
source activate informer-env
srun -u python hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features M --seq_len 672 --label_len 96 --pred_len 24 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5
scrun -u python hp_search.py --model informer --checkpoints ${2} --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5  
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5  


