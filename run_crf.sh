#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module add cuda/9.0
module add cudnn/7-cuda-9.0
source $HOME/summerResearch/env/bin/activate
python3 alined.py bpe_hindi_train_3.pt 1 hindi_1