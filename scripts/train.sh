#!/bin/bash
set -eo pipefail

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-00:10:00
#SBATCH --output=%N-%j.out

EPOCHS=2
DATA_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--data-directory)
      DATA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-directory)
      OUTPUT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      shift # past argument
      ;;
  esac
done

# Find latetst Singularity module on Compute Canada by running 'module spider singularity'
module load singularity/3.8

# #
# # Pipe output to another file
# # 
singularity exec --nv --cleanenv --pwd /code \
  --bind $DATA_PATH:/data --bind ${OUTPUT_PATH}:/output \
  train_attention.image \
    python -m segmentation.train.attention_unet \
      --epochs $EPOCHS \
      --data-directory /data \
      --output-directory /output | train_output.txt
