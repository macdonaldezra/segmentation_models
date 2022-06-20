#!/bin/bash

# if an error occurs, return exit code of the rightmost command that failed
set -eo pipefail

salloc --mem-per-cpu=2000 --cpus-per-task=4 --time=2:0:0
module load singularity/2.5

