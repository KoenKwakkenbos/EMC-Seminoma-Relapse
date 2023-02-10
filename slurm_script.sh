#!/bin/bash
#SBATCH --ntasks=12           ### How many CPU cores do you need?
#SBATCH --mem=50G            ### How much RAM memory do you need?
#SBATCH -p short           ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH -t 0-48:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/kkwakkenbos/repositories/EMC-Seminoma-Relapse/output_lr/out_%j.log        ### Where to store the console output (%j is the job number)
#SBATCH -e /trinity/home/kkwakkenbos/repositories/EMC-Seminoma-Relapse/output_lr/error_%j.log      ### Where to store the error output
#SBATCH --job-name=MIL_lr  ### Name your job so you can distinguish between jobs

# Load the modules

module purge
# automatically loads all dependencies such as cuda
# replace with required tensorflow version! (check with module avail which tensorflow version are available)
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4  

# use when you need to read/write many files quickly in tmp directory:
# source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env

# activate virtualenv after loading tensorflow/python module
# replace with your own virtualenv!
source /trinity/home/kkwakkenbos/venv_example_project/bin/activate 
echo "Hello world"

python train_MIL.py
