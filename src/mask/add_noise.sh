#!/bin/bash
#SBATCH -w ttnusa2
#SBATCH -o generate.out
#SBATCH -n 1

srun -n 1 python add_noise.py augment \
    --speech_path /data07/mayi/voxceleb_asvtorch/VoxCeleb1/cat/v1 \
    --noise_file  /data07/mayi/musan/music_noise.lst \
    --workplace /data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/mask/ \
    --snr 15 10 8 5 \
    --repeate 1 \
    --noise_type music
