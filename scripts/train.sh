#!/bin/sh

# cedar cluster

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1          # Request 2 GPU "generic resources”. You will get 2 per node.

#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=%N-%j.out

set -eo pipefail
# use latetst Singularity module on Compute Canada by running 'module spider singularity'
module load singularity/3.8

# pipe output to another file
# 
singularity run train_attention.image --bind ~/scratch/mri_data:/data --bind 