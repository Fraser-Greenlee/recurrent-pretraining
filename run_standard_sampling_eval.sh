#!/usr/bin/env bash
#SBATCH --job-name=standard_sampling_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --output=./logs/%x-%j.out
#SBATCH --requeue
#SBATCH --time=1:00:00

source .venv/bin/activate
mkdir -p logs results

python evaluate_raven/eval_standard_sampling.py \
  --model_name tomg-group-umd/huginn-0125 \
  --device cuda:0 \
  --output_file results/standard_sampling_eval.json
