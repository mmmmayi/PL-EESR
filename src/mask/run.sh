#!/bin/bash
#SBATCH -o test.out
#SBATCH --gres=gpu:1

python utils.py
#python -m torch.distributed.launch run.py --step net 