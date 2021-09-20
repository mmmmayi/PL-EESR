#!/bin/bash
#SBATCH -o output/vox2_jhu.out
#SBATCH --job-name=compa
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa4
python run_debug.py --step net
#python -m torch.distributed.launch run.py --step net 
