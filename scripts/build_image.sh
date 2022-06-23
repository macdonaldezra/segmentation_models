#!/bin/sh
# if an error occurs, return exit code of the rightmost command that failed
set -eo pipefail

ACCOUNT=$COMPUTE_CAN_ACCOUNT

# --time=days-hours:minutes:seconds
# 
salloc --mem-per-cpu=2000 --cpus-per-task=4 --time=2:20:0
module load singularity/3.8

# build on your own machine and scp it over

