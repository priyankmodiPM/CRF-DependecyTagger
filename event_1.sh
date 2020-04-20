#!/bin/bash
#SBATCH -A nlp
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

echo $1
python3 alined.py bpe_$1_train_3.pt 1 $1_1

# echo 2
# python3 emnlp_repr_combine_neuralST.py event_test_hindi_train_3.pt 3

