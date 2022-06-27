## Segmentation Models

A collection of segmentation model implementations in TensorFlow.


### Install
This project uses [Pyenv](https://github.com/pyenv/pyenv#installation) for managing the Python version and environment. To create the virtual environment and install dependencies run the following:  

```bash
pyenv virtualenv 3.8.5 segmentation-env
pyenv activate segmentation-env
pip install requirements.txt
```

### Build Singularity Container
You can build the [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/index.html) container for this project which is currently configured to train an Attention UNet model to perform image segmentation on a popular brain MRI dataset by running the following command:

```bash
# Note that Singularity will only work on Linux
sudo singularity build train.image train.def
# To train the model
sbatch --account=<account_name> scripts/run.sh --epochs 5 --data-directory <data_dir_path> --output-directory <output_dir_path>
# monitor job placement in queue
squeue -u $USER
# Get a short summary of CPU and memory efficiency of a job
seff <job_id>
# Cancel a job
scancel <job_id>
```
