#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --output=op_file.txt
source /home2/tgv2002/miniconda3/etc/profile.d/conda.sh
conda activate py37
bash run_preprocess.sh
bash fine_tune_jigsaw_attention_based.sh
