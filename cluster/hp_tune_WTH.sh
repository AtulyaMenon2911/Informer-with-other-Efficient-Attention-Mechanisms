#!/usr/bin/env bash
#SBATCH --job-name=informer_train
#SBATCH --output=informer_train%j.log
#SBATCH --error=informer_train%j.err
#SBATCH --mail-user=praphul@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
export PATH="/home/praphul/anaconda3/bin:$PATH"
source activate informer-env
srun -u python hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./  --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 --factor 3
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./ --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./ --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./  --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./  --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./  --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH  --root_path ./ --features S --seq_len 720 --label_len 168 --pred_len 48 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH --root_path ./ --features S --seq_len 720 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH --root_path ./ --features S --seq_len 720 --label_len 336 --pred_len 336 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 
srun -u python  hp_search.py --model informer  --checkpoints ${2} --data WTH --root_path ./ --features S --seq_len 720 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn ${1} --des 'Exp' --itr 5 