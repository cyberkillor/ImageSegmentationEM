#!/bin/bash

#SBATCH --job-name=ImageSeg
#SBATCH --output=log/ImageSeg_%A_%a.out
#SBATCH --error=log/ImageSeg_%A_%a.err
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1

# Print the task id.
module load cuda
conda activate torch_py3.6.3
python main.py --result_path=./result/AttU_Net700 --model_type=AttU_Net --epoch=700
